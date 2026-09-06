"""Bead i6v probe: one whole gated-delta layer of Qwen3.8-Flash-Next at S=1 as
ONE program, weights as int4 palettes (the production byte count), the
recurrent state as a constant. The question is the per-call time of a
whole-layer dispatch and which ops the plan keeps on the ANE, not numerics,
so the sidecar carries the input only (the harness times without a reference).
Shapes: hidden 2560, in_proj_qkv 10240 (q 2048, k 2048, v 6144), z 6144,
a/b 48 (v heads), K heads 16 x 128, V heads 48 x 128, out_proj 2560 x 6144,
shared expert 640."""
import json, os, sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as T

OUT = sys.argv[1]; name = "k_gdn_layer_S1"
rng = np.random.default_rng(11)
H, QKV, Z, NV, NK, D, INTER = 2560, 10240, 6144, 48, 16, 128, 640

def lut_w(O, K, seed):
    """int4 palette [O,K,1,1]: per-tensor fp16 LUT, uint4 indices."""
    w = (rng.standard_normal((O, K)) * 0.02).astype(np.float32)
    s = np.abs(w).max() / 8.0
    lut = np.array([(i - 8) * s for i in range(16)], dtype=np.float16)
    c = np.clip(np.round(w / s) + 8, 0, 15).astype(T.np_uint4_dtype)
    return c.reshape(O, K, 1, 1), lut.reshape(1, 1, 1, 1, 16, 1)

def conv(x, O, K, tag):
    c, lut = lut_w(O, K, tag)
    w = mb.constexpr_lut_to_dense(indices=c, lut=lut)
    return mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1, name=tag)

def rms(x, name):  # over the channel axis of [1,C,1,1]
    ms = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[1], keep_dims=True)
    return mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=ms, y=np.float16(1e-6))), name=name)

@mb.program(input_specs=[mb.TensorSpec(shape=(1, H, 1, 1), dtype=T.fp16)], opset_version=ct.target.iOS18)
def prog(x):
    xn = rms(x, "norm_in")
    qkv = conv(xn, QKV, H, "in_proj_qkv")            # [1,10240,1,1]
    z = conv(xn, Z, H, "in_proj_z")                  # [1,6144,1,1]
    a = conv(xn, NV, H, "in_proj_a")                 # [1,48,1,1]
    b = conv(xn, NV, H, "in_proj_b")
    # conv1d over a 4-wide causal window: the 3 cached rows are constants here; a
    # depthwise conv with kernel [1,4] over [1,QKV,1,4].
    win = mb.concat(values=[np.zeros((1, QKV, 1, 3), dtype=np.float16), qkv], axis=3)
    dw = (rng.standard_normal((QKV, 1, 1, 4)) * 0.3).astype(np.float16)
    conv_out = mb.silu(x=mb.conv(x=win, weight=dw, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=QKV), name="conv1d")  # [1,QKV,1,1]
    q = mb.slice_by_index(x=conv_out, begin=[0, 0, 0, 0], end=[1, 2048, 1, 1])
    k = mb.slice_by_index(x=conv_out, begin=[0, 2048, 0, 0], end=[1, 4096, 1, 1])
    v = mb.slice_by_index(x=conv_out, begin=[0, 4096, 0, 0], end=[1, 10240, 1, 1])
    # heads: q,k [16,128] -> repeated to 48 v-heads (GQA 3:1); v [48,128]
    qh = mb.reshape(x=q, shape=[NK, 1, D]); kh = mb.reshape(x=k, shape=[NK, 1, D]); vh = mb.reshape(x=v, shape=[NV, D, 1])
    qh = mb.concat(values=[qh, qh, qh], axis=0); kh = mb.concat(values=[kh, kh, kh], axis=0)   # [48,1,128]
    # gates: g = exp(-softplus(a + dt_bias) * exp(aLog)); beta = sigmoid(b)
    dt_bias = (rng.standard_normal((1, NV, 1, 1)) * 0.1).astype(np.float16); alog = (rng.standard_normal((1, NV, 1, 1)) * 0.1).astype(np.float16)
    g = mb.exp(x=mb.mul(x=mb.mul(x=mb.softplus(x=mb.add(x=a, y=dt_bias)), y=mb.exp(x=alog)), y=np.float16(-1.0)))  # [1,48,1,1]
    beta = mb.sigmoid(x=b)
    g3 = mb.reshape(x=g, shape=[NV, 1, 1]); beta3 = mb.reshape(x=beta, shape=[NV, 1, 1])
    state = (rng.standard_normal((NV, D, D)) * 0.05).astype(np.float16)     # S [48,128(k),128(v)] as a constant
    # delta rule at one row: kS = k @ S -> [48,1,128]; delta = beta*(v^T - kS); S' = g*S + k^T @ delta
    kS = mb.matmul(x=kh, y=state)                                            # [48,1,128]
    vT = mb.transpose(x=vh, perm=[0, 2, 1])                                  # [48,1,128]
    delta = mb.mul(x=mb.sub(x=vT, y=kS), y=beta3)                            # [48,1,128]
    kT = mb.transpose(x=kh, perm=[0, 2, 1])                                  # [48,128,1]
    s_new = mb.add(x=mb.mul(x=state, y=g3), y=mb.matmul(x=kT, y=delta), name="state_out")   # [48,128,128]
    o = mb.matmul(x=qh, y=s_new)                                             # [48,1,128]
    o4 = mb.reshape(x=o, shape=[1, NV * D, 1, 1])
    zg = mb.mul(x=z, y=mb.sigmoid(x=z))                                      # gated norm's gate (silu form)
    o4 = mb.mul(x=rms(o4, "norm_out"), y=zg)
    attn = conv(o4, H, NV * D, "out_proj")                                   # [1,2560,1,1]
    h1 = mb.add(x=x, y=attn)
    hn = rms(h1, "norm_mlp")
    gate = conv(hn, INTER, H, "se_gate"); up = conv(hn, INTER, H, "se_up")
    act = mb.mul(x=mb.real_div(x=gate, y=mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1.0))), y=np.float16(1.0))), y=up)
    se = conv(act, H, INTER, "se_down")
    y = mb.add(x=h1, y=se, name="y")
    return y, s_new

m = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
pkg = os.path.join(OUT, name + ".mlpackage"); m.save(pkg)
side = os.path.join(OUT, name + ".probe"); os.makedirs(side, exist_ok=True)
x = rng.standard_normal((1, H)).astype(np.float16); x.tofile(os.path.join(side, "x_f16.bin"))
json.dump({"O": H, "K": H, "S": 1, "input": "x", "timing_only": True}, open(os.path.join(side, "meta.json"), "w"))
print("built", name, "outputs", list(m.output_description) if hasattr(m, "output_description") else "?")
