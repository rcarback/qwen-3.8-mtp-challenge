"""Build .mlpackage probes for the ANE compute-plan / numerics / timing test.

Every weight form is applied to the SAME fp32 weight so the arms are
comparable; each package gets a sidecar dir <name>.probe/ holding the
effective fp16 weight the program is meant to compute with, the fp16 input,
and an fp32 numpy reference y = x @ w_eff.T, so the Swift side compares
without re-deriving any quantizer.
"""
import json, os, sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as T

OUT = sys.argv[1]
O, K, S = 1280, 2560, 256
rng = np.random.default_rng(1)
w = (rng.standard_normal((O, K)) * 0.02).astype(np.float32)
x = (rng.standard_normal((S, K))).astype(np.float16)

def mlx_affine(w, g, bits):
    """MLX affine quantization along the input axis: per group of g,
    scale=(max-min)/(2^bits-1), bias=min, q=round((w-bias)/scale) in [0,2^bits-1].
    Returns q (uint), scale fp16 [O,K/g], bias fp16 [O,K/g], and the effective
    fp32 weight the ANE form scale*(q-offset) reconstructs with offset=-bias/scale in fp16."""
    Oo, Kk = w.shape
    wg = w.reshape(Oo, Kk // g, g)
    mx = wg.max(-1, keepdims=True); mn = wg.min(-1, keepdims=True)
    qmax = 2 ** bits - 1
    scale = (mx - mn) / qmax
    scale = np.where(scale == 0, 1e-6, scale)
    q = np.clip(np.round((wg - mn) / scale), 0, qmax)
    scale16 = scale.astype(np.float16); bias16 = mn.astype(np.float16)
    offset16 = (-bias16.astype(np.float32) / scale16.astype(np.float32)).astype(np.float16)
    w_eff = (scale16.astype(np.float32) * (q - offset16.astype(np.float32))).reshape(Oo, Kk)
    return q.reshape(Oo, Kk), scale16.reshape(Oo, Kk // g), offset16.reshape(Oo, Kk // g), w_eff.astype(np.float32)

def per_channel_int8(w):
    s = (np.abs(w).max(1, keepdims=True) / 127.0).astype(np.float16)
    q = np.clip(np.round(w / s.astype(np.float32)), -127, 127).astype(np.int8)
    return q, s.reshape(-1), (q.astype(np.float32) * s.astype(np.float32))

def lut_uniform(w, per=None):
    """Per-tensor (per=None) or per-block-of-rows (per=int rows) uniform 16-level palette."""
    Oo, Kk = w.shape
    if per is None:
        blocks = [(0, Oo)]
    else:
        blocks = [(i, min(i + per, Oo)) for i in range(0, Oo, per)]
    codes = np.zeros((Oo, Kk), dtype=np.uint8); luts = []; w_eff = np.zeros_like(w)
    for (a, b) in blocks:
        s = np.abs(w[a:b]).max() / 8.0
        lut = np.array([(i - 8) * s for i in range(16)], dtype=np.float16)
        c = np.clip(np.round(w[a:b] / s) + 8, 0, 15).astype(np.uint8)
        codes[a:b] = c; luts.append(lut); w_eff[a:b] = lut.astype(np.float32)[c]
    return codes, np.stack(luts), w_eff

def lut_pcs(w):
    """Per-channel scale + shared int8 codebook: w ~ s[o] * lut[code], lut int8 uniform in [-8,7]*16."""
    s = (np.abs(w).max(1, keepdims=True) / 8.0).astype(np.float16)     # per-channel scale
    c = np.clip(np.round(w / s.astype(np.float32)) + 8, 0, 15).astype(np.uint8)
    lut8 = np.array([(i - 8) for i in range(16)], dtype=np.int8)          # int8 codebook
    w_eff = (s.astype(np.float32) * lut8.astype(np.float32)[c])
    return c, lut8, s.reshape(O, 1, 1, 1), w_eff

def conv_prog(weight_fn, opset=ct.target.iOS18):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, K, 1, S), dtype=T.fp16)], opset_version=opset)
    def prog(x):
        wt = weight_fn()
        return mb.conv(x=x, weight=wt, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1)
    return prog

forms = {}
w16 = w.astype(np.float16)
forms["a_fp16"] = (lambda: mb.const(val=w16.reshape(O, K, 1, 1)), w16.astype(np.float32))

q8, s8, w8 = per_channel_int8(w)
forms["b_int8_perchannel_affine"] = (lambda: mb.constexpr_affine_dequantize(
    quantized_data=q8.reshape(O, K, 1, 1), zero_point=np.int8(0), scale=s8, axis=np.int32(0)), w8)

q, sc, of, we = mlx_affine(w, 32, 8)
forms["c_int8_blockwise_g32"] = (lambda q=q, sc=sc, of=of: mb.constexpr_blockwise_shift_scale(
    data=q.astype(np.uint8).reshape(O, K, 1, 1), scale=sc.reshape(O, K // 32, 1, 1), offset=of.reshape(O, K // 32, 1, 1)), we)

q, sc, of, we = mlx_affine(w, 32, 4)
forms["d_int4_blockwise_g32"] = (lambda q=q, sc=sc, of=of: mb.constexpr_blockwise_shift_scale(
    data=q.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), scale=sc.reshape(O, K // 32, 1, 1), offset=of.reshape(O, K // 32, 1, 1)), we)

q, sc, of, we = mlx_affine(w, 64, 4)
forms["e_int4_blockwise_g64"] = (lambda q=q, sc=sc, of=of: mb.constexpr_blockwise_shift_scale(
    data=q.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), scale=sc.reshape(O, K // 64, 1, 1), offset=of.reshape(O, K // 64, 1, 1)), we)

# int4 blockwise WITHOUT offset (symmetric), to separate "op rejected" from "offset rejected".
q, sc, of, we = mlx_affine(w, 32, 4)
q_sym = (q.astype(np.int32) - 8).astype(T.np_int4_dtype)
we_sym = (sc.astype(np.float32).repeat(32, axis=1) * (q.astype(np.float32) - 8)).astype(np.float32)
forms["f_int4_blockwise_g32_nooffset"] = (lambda q_sym=q_sym, sc=sc: mb.constexpr_blockwise_shift_scale(
    data=q_sym.reshape(O, K, 1, 1), scale=sc.reshape(O, K // 32, 1, 1)), we_sym)

c, luts, we = lut_uniform(w)
forms["g_int4_lut_pertensor"] = (lambda c=c, luts=luts: mb.constexpr_lut_to_dense(
    indices=c.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=luts.reshape(1, 1, 1, 1, 16, 1)), we)

c, luts, we = lut_uniform(w, per=64)
forms["h_int4_lut_per64rows"] = (lambda c=c, luts=luts: mb.constexpr_lut_to_dense(
    indices=c.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=luts.reshape(O // 64, 1, 1, 1, 16, 1)), we)

c, lut8, s, we = lut_pcs(w)
def _pcs(c=c, lut8=lut8, s=s):
    d = mb.constexpr_lut_to_dense(indices=c.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=lut8.reshape(1, 1, 1, 1, 16, 1))
    return mb.constexpr_blockwise_shift_scale(data=d, scale=s)
forms["i_int4_lut_int8codebook_perchannel_scale"] = (_pcs, we)

# LUT fp16 per-tensor followed by a RUNTIME per-channel mul (the fallback if the chained constexpr is refused).
c, luts, we0 = lut_uniform(w)
s_pc = (np.ones((O, 1, 1, 1), dtype=np.float16))
def _lut_mul(c=c, luts=luts):
    d = mb.constexpr_lut_to_dense(indices=c.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=luts.reshape(1, 1, 1, 1, 16, 1))
    return mb.mul(x=d, y=s_pc)
forms["j_int4_lut_runtime_mul"] = (_lut_mul, we0)

x32 = x.astype(np.float32)
manifest = {}
for name, (fn, w_eff) in forms.items():
    try:
        m = ct.convert(conv_prog(fn), minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
        pkg = os.path.join(OUT, name + ".mlpackage"); m.save(pkg)
        side = os.path.join(OUT, name + ".probe"); os.makedirs(side, exist_ok=True)
        w_eff.astype(np.float16).tofile(os.path.join(side, "w_eff_f16.bin"))
        x.tofile(os.path.join(side, "x_f16.bin"))
        (x32 @ w_eff.astype(np.float32).T).astype(np.float32).tofile(os.path.join(side, "y_ref_f32.bin"))
        json.dump({"O": O, "K": K, "S": S, "input": "x", "output": list(m.output_description.keys())[0] if hasattr(m, "output_description") else None},
                  open(os.path.join(side, "meta.json"), "w"))
        manifest[name] = "ok"
        print("built", name)
    except Exception as e:
        manifest[name] = "convert failed: " + str(e)[:300]
        print("FAILED", name, str(e)[:300])

# Misc op eligibility program: several independent outputs from one [1,2560,1,256] input.
def misc():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, K, 1, S), dtype=T.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        outs = []
        outs.append(mb.silu(x=x, name="silu"))
        outs.append(mb.gelu(x=x, name="gelu"))
        outs.append(mb.sigmoid(x=x, name="sigmoid"))
        outs.append(mb.tanh(x=x, name="tanh"))
        outs.append(mb.exp(x=x, name="exp"))
        outs.append(mb.softmax(x=x, axis=1, name="softmax_c"))
        outs.append(mb.layer_norm(x=x, axes=[1], name="layer_norm_c"))
        outs.append(mb.reduce_mean(x=x, axes=[1], keep_dims=True, name="reduce_mean_c"))
        outs.append(mb.rsqrt(x=mb.add(x=mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[1], keep_dims=True), y=np.float16(1e-6)), name="rms_rsqrt"))
        outs.append(mb.sqrt(x=mb.abs(x=x), name="sqrt"))
        xt = mb.reshape(x=x, shape=[1, K, S], name="reshape_3d")
        xt = mb.transpose(x=xt, perm=[0, 2, 1], name="transpose_sk")          # [1,S,K]
        outs.append(mb.matmul(x=xt, y=(rng.standard_normal((K, 640)) * 0.02).astype(np.float16), name="matmul_const"))
        outs.append(mb.linear(x=xt, weight=(rng.standard_normal((640, K)) * 0.02).astype(np.float16), name="linear_const"))
        q = mb.reshape(x=xt, shape=[1, S, 10, 256], name="q4d"); q = mb.transpose(x=q, perm=[0, 2, 1, 3], name="q_heads")  # [1,10,S,256]
        outs.append(mb.scaled_dot_product_attention(query=q, key=q, value=q, name="sdpa"))
        outs.append(mb.reduce_max(x=x, axes=[1], keep_dims=True, name="reduce_max_c"))
        outs.append(mb.clip(x=x, alpha=np.float16(-1.0), beta=np.float16(1.0), name="clip"))
        outs.append(mb.erf(x=x, name="erf"))
        outs.append(mb.concat(values=[x, x], axis=1, name="concat_c"))
        outs.append(mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, 1280, 1, S], name="slice_c"))
        return tuple(outs)
    return prog
try:
    m = ct.convert(misc(), minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(os.path.join(OUT, "z_misc_ops.mlpackage")); manifest["z_misc_ops"] = "ok"; print("built z_misc_ops")
except Exception as e:
    manifest["z_misc_ops"] = "convert failed: " + str(e)[:400]; print("FAILED z_misc_ops", str(e)[:400])
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
