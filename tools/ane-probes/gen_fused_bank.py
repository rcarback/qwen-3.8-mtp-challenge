"""Bead 8xv: the dense tower's fused SwiGLU-down ANE prefix as ONE multifunction
.mlpackage per bucket, all layers as functions, int4 palette weights with a
16-entry codebook PER ROW (fitted by Lloyd iterations on that row), the form
the compute plan confirmed on the ANE at no time cost over a per-tensor LUT.

usage: gen_fused_bank.py WEIGHTS_DIR OUT_DIR FRACTION BUCKET [LAYERS e.g. 0-63 or 0,1]
Reads the MLX q4 group-64 safetensors directly (uint32 packed nibbles, low
nibble first; bf16 scales and biases per group along the input axis).
Output: OUT_DIR/S<BUCKET>.mlpackage with functions layer<N> (PART=k in the
environment names it S<BUCKET>.p<k>, one part of a bank split by layer range); input x
[1,hidden,1,S] fp16, output y [1,hidden,1,S] fp16 = down(silu(gate x) * up x)
over the first F intermediate channels, F = round(FRACTION*inter/64)*64."""
import json, os, struct, sys, shutil, time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as T
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

W, OUT, FRAC, S = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
spec = sys.argv[5] if len(sys.argv) > 5 else "0-63"
# Rows per codebook. 1 = one codebook per row (lossless, but the ANE compiler
# rejects the program and Core ML runs it on the GPU); 64 = one per 64 rows
# (ANE-eligible in the compute plan). Set ROWS_PER_LUT in the environment.
ROWS_PER_LUT = int(os.environ.get("ROWS_PER_LUT", "1"))
layers = list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1)) if "-" in spec else [int(x) for x in spec.split(",")]
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(os.path.join(W, "model.safetensors.index.json")))["weight_map"]
_hdr = {}
def tensor(name):
    f = idx[name]; path = os.path.join(W, f)
    if f not in _hdr:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; _hdr[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = _hdr[f]; info = h[name]; a, b = info["data_offsets"]
    m = np.memmap(path, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
    if info["dtype"] == "U32": return np.frombuffer(m, dtype=np.uint32).reshape(info["shape"])
    if info["dtype"] == "BF16":
        u = np.frombuffer(m, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(info["shape"])
    raise ValueError(info["dtype"])

def dequant(prefix, rows=None, cols=None):
    """MLX affine q4 g64 -> fp32 [O, K] (optionally a row or column slice)."""
    w = tensor(prefix + ".weight"); s = tensor(prefix + ".scales"); b = tensor(prefix + ".biases")
    if rows is not None: w, s, b = w[:rows], s[:rows], b[:rows]
    O, Kp = w.shape; K = Kp * 8
    if cols is not None:
        w, s, b = w[:, : cols // 8], s[:, : cols // 64], b[:, : cols // 64]; K = cols
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    q = ((w[:, :, None] >> shifts[None, None, :]) & 0xF).astype(np.float32).reshape(O, K)
    g = np.repeat(s, 64, axis=1)[:, :K]; bb = np.repeat(b, 64, axis=1)[:, :K]
    return g * q + bb

def row_codebooks(w, iters=8, chunk=256):
    """Per-row-block 16-level Lloyd quantizer (ROWS_PER_LUT rows share a codebook).
    Returns codes uint8 [O,K] and lut fp16 [O/ROWS_PER_LUT,16]."""
    if ROWS_PER_LUT > 1:
        O, K = w.shape
        assert O % ROWS_PER_LUT == 0
        wb = w.reshape(O // ROWS_PER_LUT, ROWS_PER_LUT * K)
        codes_b, luts_b = row_codebooks_rows(wb, iters, chunk)
        return codes_b.reshape(O, K), luts_b
    return row_codebooks_rows(w, iters, chunk)

def row_codebooks_rows(w, iters=8, chunk=256):
    O, K = w.shape
    codes = np.empty((O, K), dtype=np.uint8); luts = np.empty((O, 16), dtype=np.float16)
    qs = (np.arange(16) + 0.5) / 16.0
    for a in range(0, O, chunk):
        wc = w[a:a + chunk].astype(np.float32)                      # [c,K]
        c = np.quantile(wc, qs, axis=1).T.astype(np.float32)        # [c,16] init from row quantiles
        for _ in range(iters):
            mids = (c[:, :-1] + c[:, 1:]) / 2                        # [c,15]
            code = (wc[:, :, None] > mids[:, None, :]).sum(-1)       # [c,K] in 0..15
            for j in range(16):
                m = code == j
                cnt = m.sum(1); sm = (wc * m).sum(1)
                c[:, j] = np.where(cnt > 0, sm / np.maximum(cnt, 1), c[:, j])
            c.sort(axis=1)
        mids = (c[:, :-1] + c[:, 1:]) / 2
        codes[a:a + chunk] = (wc[:, :, None] > mids[:, None, :]).sum(-1).astype(np.uint8)
        luts[a:a + chunk] = c.astype(np.float16)
    return codes, luts

def layer_program(hidden, F, gc, gl, uc, ul, dc, dl):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, hidden, 1, S), dtype=T.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        gw = mb.constexpr_lut_to_dense(indices=gc.astype(T.np_uint4_dtype).reshape(F, hidden, 1, 1), lut=gl.reshape(F // ROWS_PER_LUT, 1, 1, 1, 16, 1))
        uw = mb.constexpr_lut_to_dense(indices=uc.astype(T.np_uint4_dtype).reshape(F, hidden, 1, 1), lut=ul.reshape(F // ROWS_PER_LUT, 1, 1, 1, 16, 1))
        dw = mb.constexpr_lut_to_dense(indices=dc.astype(T.np_uint4_dtype).reshape(hidden, F, 1, 1), lut=dl.reshape(hidden // ROWS_PER_LUT, 1, 1, 1, 16, 1))
        gate = mb.conv(x=x, weight=gw, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1, name="gate")
        up = mb.conv(x=x, weight=uw, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1, name="up")
        # SiLU spelled as x / (1 + exp(-x)): the ANE's silu op is a coarse table.
        if os.environ.get("SILU_OP") == "native":
            act = mb.mul(x=mb.silu(x=gate), y=up, name="swiglu")   # placement probe: the single silu op
        else:
            den = mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1.0))), y=np.float16(1.0))
            act = mb.mul(x=mb.real_div(x=gate, y=den), y=up, name="swiglu")
        return mb.conv(x=act, weight=dw, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1, name="y")
    return prog

tmp = os.path.join(OUT, "tmp-S%d" % S); os.makedirs(tmp, exist_ok=True)
desc = MultiFunctionDescriptor(); meta = {"fraction": FRAC, "bucket": S, "layers": [], "form": "int4-lut-per-%d-rows" % ROWS_PER_LUT}
for n in layers:
    t0 = time.time(); p = "language_model.model.layers.%d.mlp." % n
    inter = tensor(p + "gate_proj.scales").shape[0]; hidden = tensor(p + "gate_proj.weight").shape[1] * 8
    F = int(round(FRAC * inter / 64)) * 64
    pkg = os.path.join(tmp, "layer%d.mlpackage" % n)
    if os.path.isdir(os.path.join(pkg, "Data")):
        # Resume: a per-layer package from an earlier run of this bucket is reused as is.
        desc.add_function(pkg, src_function_name="main", target_function_name="layer%d" % n)
        meta["layers"].append({"layer": n, "F": F, "hidden": hidden, "resumed": True})
        print("layer %d reused" % n, flush=True); continue
    g = dequant(p + "gate_proj", rows=F); u = dequant(p + "up_proj", rows=F); d = dequant(p + "down_proj", cols=F)
    gc, gl = row_codebooks(g); uc, ul = row_codebooks(u); dc, dl = row_codebooks(d)
    def err(w, c, l):
        lut_rows = np.repeat(l.astype(np.float32), ROWS_PER_LUT, axis=0)   # [O,16]
        return float(np.abs(np.take_along_axis(lut_rows, c.astype(np.int64), 1) - w).mean() / np.abs(w).mean())
    m = ct.convert(layer_program(hidden, F, gc, gl, uc, ul, dc, dl), minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    m.save(pkg)
    desc.add_function(pkg, src_function_name="main", target_function_name="layer%d" % n)
    meta["layers"].append({"layer": n, "F": F, "hidden": hidden, "rel_err_gate": err(g, gc, gl), "rel_err_down": err(d, dc, dl)})
    print("layer %d F=%d gate relerr %.4f down relerr %.4f  %.1fs" % (n, F, meta["layers"][-1]["rel_err_gate"], meta["layers"][-1]["rel_err_down"], time.time() - t0), flush=True)
desc.default_function_name = "layer%d" % layers[0]
STEM = "S%d" % S + (".p%s" % os.environ["PART"] if os.environ.get("PART") else "")   # PART=k names a part: S<bucket>.p<k>
out = os.path.join(OUT, STEM + ".mlpackage")
if os.path.exists(out): shutil.rmtree(out)
save_multifunction(desc, out)
json.dump(meta, open(os.path.join(OUT, STEM + ".json"), "w"), indent=1)
shutil.rmtree(tmp)
print("saved", out, flush=True)
