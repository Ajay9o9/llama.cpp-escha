#!/usr/bin/env python3
"""Generate golden test cases for GGML_OP_ESCHA_LINEAR into a GGUF.

Reference math mirrors escha_mlx.ref.py + the deploy TCQEschaLinear forward:
  xh   = f16(H128(x * s_in * rin) * RS)
  mid  = xh_f32 @ decode(code)_f32
  y    = f16(H128(mid) * RS * rout) * s_out + bias
"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

M32 = 0xFFFFFFFF
RS = 0.088388347648


def cba_lut():
    x = np.arange(65536, dtype=np.uint64)
    x = (x * np.uint64(0xCBAC1FED)) & np.uint64(M32)
    x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = ((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float16)


_LUT = cba_lut()


def lane_positions(lane):
    l0 = lane & ~4
    c_off = (lane >> 2) & 1
    pos = []
    for j in range(8):
        fi = j >> 1
        row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8
        col = 2 * ((l0 >> 3) + (4 if j >= 4 else 0)) + c_off
        pos.append((row, col))
    return pos


_POS = None
def positions():
    global _POS
    if _POS is None:
        p = np.empty(256, dtype=np.int64)
        for lane in range(32):
            for j, (r, c) in enumerate(lane_positions(lane)):
                p[lane * 8 + j] = r * 16 + c
        _POS = p
    return _POS


def k2_lane_consts():
    i0s, i1s, shifts = [], [], []
    for lane in range(32):
        t_off = lane * 8
        i1 = t_off >> 4
        i0s.append((i1 + 15) & 15)
        i1s.append(i1)
        shifts.append(((~t_off) & 8) << 1)
    return np.array(i0s), np.array(i1s), np.array(shifts, dtype=np.uint64)


_K2 = k2_lane_consts()


def decode_states_k2(words):
    i0s, i1s, shifts = _K2
    merged = (words[:, i0s].astype(np.uint64) << 32) | words[:, i1s].astype(np.uint64)
    w = (merged >> shifts) & np.uint64(M32)
    js = np.array([2 * (7 - j) for j in range(8)], dtype=np.uint64)
    return (w[:, :, None] >> js) & np.uint64(0xFFFF)


def h128(x):
    shape = x.shape
    n = shape[-1]
    y = x.astype(np.float32).reshape(-1, n // 128, 128).copy()
    h = 1
    while h < 128:
        y = y.reshape(-1, n // 128, 128 // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
    return y.reshape(shape)


def reconstruct_fast(code_u16, ic, oc):
    tk, tn = ic // 16, oc // 16
    words = np.ascontiguousarray(code_u16.reshape(tk * tn, 32).astype(np.uint16)).view(np.uint32)
    states = decode_states_k2(words)                       # (T, 32, 8)
    vals = _LUT[states.astype(np.uint16).reshape(words.shape[0], 256)]
    tiles = np.zeros((words.shape[0], 256), dtype=np.float16)
    tiles[:, positions()] = vals
    return (tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3)
            .reshape(ic, oc).copy())


def make_case(rng, ic, oc, m, k=2):
    assert k == 2
    n_code = 16 * k
    code = rng.integers(0, 2**16, size=(ic // 16) * (oc // 16) * n_code, dtype=np.uint16)
    w_bare = reconstruct_fast(code, ic, oc).astype(np.float32)
    rin = rng.normal(1.0, 0.05, ic).astype(np.float16)
    rout = rng.normal(1.0, 0.05, oc).astype(np.float16)
    s_in = (rng.normal(1.0, 0.02, ic)).astype(np.float32)
    s_out = (rng.normal(1.0, 0.02, oc)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, oc).astype(np.float16)
    x = rng.normal(0.0, 1.0, (m, ic)).astype(np.float32)

    xh = (h128(x * s_in[None, :].astype(np.float32) * rin[None, :].astype(np.float32)) * RS).astype(np.float16)
    mid = xh.astype(np.float32) @ w_bare
    y = (h128(mid) * RS * rout[None, :].astype(np.float32)).astype(np.float16)
    y = (y.astype(np.float32) * s_out[None, :] + bias[None, :].astype(np.float32)).astype(np.float32)

    return {
        "escha.code":  code.view(np.int16),
        "escha.rin":   rin,
        "escha.rout":  rout,
        "escha.s_in":  s_in,
        "escha.s_out": s_out,
        "escha.bias":  bias,
        "escha.x":     x,
        "escha.yref":  y,
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "escha-linear-cases.gguf"
    rng = np.random.default_rng(42)
    writer = gguf.GGUFWriter(out, "escha_linear_cases")

    cases = [
        ("c512_o768_m3", 512, 768, 3),
        ("c256_o384_m1", 256, 384, 1),
        ("c1024_o1152_m5", 1024, 1152, 5),
    ]
    for name, ic, oc, m in cases:
        t = make_case(rng, ic, oc, m)
        for key, arr in t.items():
            dtype = gguf.GGMLQuantizationType.I16 if arr.dtype == np.int16 else (
                    gguf.GGMLQuantizationType.F16 if arr.dtype == np.float16 else
                    gguf.GGMLQuantizationType.F32)
            writer.add_tensor(f"{name}.{key}", arr, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
