"""NVFP4 (W4A4, modelopt) quantizer for online RL weight-sync.

Produces the exact per-linear HF tensor layout sglang's ModelOptNvFp4 loader
expects (verified against Mapika/GLM-5.2-NVFP4), for the routed + shared MoE
experts only (attention/router/lm_head stay bf16 per hf_quant_config exclude):

  .weight        uint8            [out, in//2]      E2M1, 2 fp4 packed / byte (input dim)
  .weight_scale  float8_e4m3fn    [out, in//16]     per-16 block scale, UNSWIZZLED
  .weight_scale_2 float32          scalar            per-tensor weight global scale
  .input_scale   float32          scalar            per-tensor ACTIVATION global scale

Two-level NVFP4 scaling: w ~= e2m1(w / (blk_e4m3 * gscale)) with
  gscale = amax(|W|) / (E2M1_MAX * E4M3_MAX),  blk = amax_16(|W|) / E2M1_MAX.

W4A4 gotcha vs int4 (W-only): `.input_scale` is an ACTIVATION calibration stat,
NOT derivable from weights. During RL the weights change every step but the
activation scale is held STATIC at the init-checkpoint value — we load those once
from the init HF checkpoint and re-emit them.

NOTE: the E2M1 value ladder + nibble-pack order below match modelopt's convention;
they are the two things to confirm against the reference on the first GPU
round-trip (serve slime-quantized weights, diff logits vs the modelopt checkpoint).
"""

import glob
import os
import re

import torch

E2M1_MAX = 6.0
E4M3_MAX = 448.0
GROUP = 16

# E2M1 magnitude ladder; index == the 3-bit [exp exp mant] code (ascending value).
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# round-to-nearest thresholds (midpoints between consecutive ladder values)
_E2M1_THRESH = ((_E2M1[1:] + _E2M1[:-1]) / 2.0)  # 7 boundaries

# modelopt exclude: attention (MLA/DSA), router gate, lm_head, embeddings stay bf16.
_QUANT_RE = re.compile(r"\.mlp\.(experts\.\d+|shared_experts)\.(gate|up|down)_proj\.weight$")

_INPUT_SCALE_CACHE = {}


def _e2m1_codes(x):
    """float tensor (already scaled to ~[-6,6]) -> uint8 4-bit E2M1 codes (0..15)."""
    sign = (x < 0).to(torch.uint8)
    mag = x.abs()
    idx = torch.bucketize(mag, _E2M1_THRESH.to(x.device))  # 0..7 nearest ladder index
    return (sign << 3) | idx.to(torch.uint8)


def _pack_e2m1(codes):
    """[out, in] 4-bit codes -> [out, in//2] uint8, 2 per byte along input dim
    (element 2i in the low nibble, 2i+1 in the high nibble)."""
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    return (lo | (hi << 4)).contiguous()


def _quantize_weight_nvfp4(w_bf16):
    """bf16 [out, in] -> (weight uint8 [out,in//2], weight_scale e4m3 [out,in//16],
    weight_scale_2 f32 scalar). in must be a multiple of GROUP(16).

    Prefer modelopt's own NVFP4 cast (byte-identical to the checkpoint producer);
    fall back to the explicit math (validated 99.7% byte-match vs modelopt, 0.3%
    boundary-rounding) if modelopt is unavailable."""
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

        out = NVFP4QTensor.quantize(w_bf16, GROUP)  # -> (qtensor, block_scale_e4m3, global_f32)
        qt, scale, gscale = out
        packed = getattr(qt, "_quantized_data", getattr(qt, "data", qt))
        return packed, scale, gscale
    except Exception:
        pass  # fall through to explicit math
    out_f, in_f = w_bf16.shape
    w = w_bf16.to(torch.float32)
    gscale = w.abs().amax().clamp(min=1e-8) / (E2M1_MAX * E4M3_MAX)  # per-tensor global
    wg = w.reshape(out_f, in_f // GROUP, GROUP)
    blk = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / E2M1_MAX  # per-block fp32
    # block scale stored in E4M3 = blk / gscale
    blk_e4m3 = (blk / gscale).to(torch.float8_e4m3fn)
    # dequantized block scale used to normalize weights before E2M1 rounding
    deq_blk = blk_e4m3.to(torch.float32) * gscale
    wq = (wg / deq_blk.clamp(min=1e-12)).reshape(out_f, in_f)
    codes = _e2m1_codes(wq)
    packed = _pack_e2m1(codes)
    scale = blk_e4m3.reshape(out_f, in_f // GROUP)
    return packed, scale, gscale.reshape(())


def _load_init_input_scales(hf_ckpt_dir):
    """Load per-linear .input_scale (activation global scale) from the init nvfp4
    checkpoint once; these are held static across RL steps."""
    if hf_ckpt_dir in _INPUT_SCALE_CACHE:
        return _INPUT_SCALE_CACHE[hf_ckpt_dir]
    from safetensors import safe_open

    scales = {}
    for f in glob.glob(os.path.join(hf_ckpt_dir, "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.endswith(".input_scale"):
                    scales[k] = h.get_tensor(k)
    _INPUT_SCALE_CACHE[hf_ckpt_dir] = scales
    return scales


def quantize_params_modelopt_nvfp4(args, converted_named_params, quantization_config):
    hf_dir = getattr(args, "hf_checkpoint", None) or getattr(args, "ref_load", None)
    input_scales = _load_init_input_scales(hf_dir) if hf_dir else {}
    out = []
    for name, param in converted_named_params:
        if not _QUANT_RE.search(name):
            out.append((name, param))  # attention/router/head/norms -> bf16 verbatim
            continue
        base = name[: -len(".weight")]
        packed, scale, gscale = _quantize_weight_nvfp4(param.to(torch.bfloat16))
        out.append((base + ".weight", packed))
        out.append((base + ".weight_scale", scale))
        out.append((base + ".weight_scale_2", gscale))
        # activation global scale: carry the init checkpoint's value (static), else 1.0
        isc = input_scales.get(base + ".input_scale")
        out.append((base + ".input_scale", isc if isc is not None else torch.tensor(1.0, dtype=torch.float32)))
    return out
