#!/usr/bin/env python3
"""Full numpy reference forward for Qwen3.8-27B-Escha-W2 (dense Escha-W2).
Token-by-token recurrent GDN + manual KV cache for full-attn layers.
Prints per-layer |h| and final top-k logits for the last position."""
import json, struct, sys, math
import numpy as np

sys.path.insert(0, '/tmp/llama-escha/tools')
import importlib.util
spec = importlib.util.spec_from_file_location("gen", "/tmp/llama-escha/tools/escha-dump-linear-cases.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)

M32 = np.uint64(0xFFFFFFFF)
CKPT_DIR = '/Volumes/medusa-1tb/models/EschaLabs/Qwen3.8-27B-Escha-W2'
SHARDS = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']

_hdrs = []
for s in SHARDS:
    f = open(f'{CKPT_DIR}/{s}', 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    _hdrs.append((f, 8 + n, {k: v for k, v in hdr.items() if k != '__metadata__'}))

DT = {'I32': np.int32, 'F32': np.float32, 'F16': np.float16, 'I16': np.int16, 'I8': np.int8}

def get(key):
    for f, base, hdr in _hdrs:
        if key in hdr:
            v = hdr[key]
            f.seek(base + v['data_offsets'][0])
            a = np.frombuffer(f.read(v['data_offsets'][1] - v['data_offsets'][0]), dtype=DT[v['dtype']])
            return a.reshape(v['shape'])
    raise KeyError(key)

# ---- codec ----
def reconstruct_k3(code_u16, ic, oc):
    tk, tn = ic // 16, oc // 16
    words = np.ascontiguousarray(code_u16.reshape(tk * tn, 48)).view(np.uint32)
    i0s, i2s, s2s = [], [], []
    for lane in range(32):
        t_off = lane * 8; b1 = (t_off + 257) * 3; b2 = b1 + 21
        i2 = ((b2 - 1) >> 5)
        i0s.append(((b1 - 16) >> 5) % 24); i2s.append(i2 % 24)
        s2s.append((((i2 + 1) << 5) - b2))
    i0s = np.array(i0s); i2s = np.array(i2s); s2s = np.array(s2s, dtype=np.uint64)
    merged = (words[:, i0s].astype(np.uint64) << 32) | words[:, i2s].astype(np.uint64)
    w7 = (merged >> s2s) & M32; w3 = (merged >> (s2s + np.uint64(12))) & M32
    sh = np.array([9, 6, 3, 0], dtype=np.uint64)
    lo = (w3[:, :, None] >> sh) & np.uint64(0xFFFF); hi = (w7[:, :, None] >> sh) & np.uint64(0xFFFF)
    states = np.concatenate([lo, hi], axis=2).astype(np.uint16)
    T = words.shape[0]
    vals = gen._LUT[states.reshape(T, 256)]
    tiles = np.zeros((T, 256), dtype=np.float16)
    tiles[:, gen.positions()] = vals
    return tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(ic, oc)

def escha_W(pre):
    cfg = get(pre + '.escha_config').tolist()
    ic, oc, K = cfg[4], cfg[5], cfg[1]
    code = get(pre + '.escha_code').view(np.uint16)
    nit, nct, n_code = ic // 16, oc // 16, 16 * K
    code = code.reshape(nit, nct, n_code)
    if K == 2:
        W = gen.reconstruct_fast(np.ascontiguousarray(code.reshape(-1)), ic, oc)
    else:
        W = reconstruct_k3(np.ascontiguousarray(code.reshape(-1)).view(np.uint16), ic, oc)
    rin = get(pre + '.escha_rin').astype(np.float32)
    rout = get(pre + '.escha_rout').astype(np.float32)
    sin_ = get(pre + '.escha_s_in').astype(np.float32)
    sout = get(pre + '.escha_s_out').astype(np.float32)
    bias = get(pre + '.bias')
    return (W, rin, rout, sin_, sout, bias)

RS = gen.RS

def escha_mm(pre, x):
    W, rin, rout, sin_, sout, bias = escha_W(pre)
    xh = (gen.h128((x * sin_) * rin) * RS).astype(np.float16)
    mid = xh.astype(np.float32) @ W.astype(np.float32)
    y = (gen.h128(mid) * RS * rout).astype(np.float16).astype(np.float32) * sout
    if bias is not None:
        y = y + bias.astype(np.float32)
    return y

# ---- tokenizer ----
tokj = json.load(open(f'{CKPT_DIR}/tokenizer.json'))
vocab = tokj['model']['vocab']
merges = [tuple(m.split()) for m in tokj['model']['merges']]
ranks = {m: i for i, m in enumerate(merges)}
tok_to_id = vocab

def bpe_encode(text):
    import re
    pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
    out = []
    for word in pat.findall(text):
        w = list(word.replace(' ', '\u0120'))
        if not w: continue
        while True:
            best, bi = None, None
            for i in range(len(w) - 1):
                r = ranks.get((w[i], w[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None: break
            w[bi:bi + 2] = [w[bi] + w[bi + 1]]
        out.extend(vocab[x] for x in w)
    return out

# ---- math helpers ----

_SHIFT_SUFFIXES = ('.input_layernorm.weight', '.post_attention_layernorm.weight',
                   '.q_norm.weight', '.k_norm.weight')
def getn(key):
    """norm weights stored as (w-1); runtime w = stored + 1 (mlx-lm sanitize quirk)"""
    w = get(key)
    if key.endswith('model.norm.weight') or any(key.endswith(s) for s in _SHIFT_SUFFIXES):
        w = w + np.float16(1.0)
    return w


import os as _os, json as _json
SUMS=[]
CUR_LAYER=[-1]
def acc(name, arr):
    if _os.environ.get('NPY_DEBUG_SUMS'):
        SUMS.append((f'{name}#{CUR_LAYER[0]}', float(np.asarray(arr,dtype=np.float64).sum())))
        if len(SUMS) % 500 == 0:
            _json.dump(SUMS, open('/tmp/npy_sums.json','w'))

def rms(x, w):
    v = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(v + 1e-6) * w.astype(np.float32)

def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))

# ---- model constants ----
NL = 64
H = 5120
NK, DK = 16, 128          # key heads, dim
NV, DV = 48, 128          # value heads
KDIM, VDIM = NK * DK, NV * DV
DT_RANK = 48
CONV_K = 4
NROT, THETA = 64, 1e7
HEADS, KVH, HDIM = 24, 4, 256
VOCAB = 248320

emb_w8 = get('model.language_model.embed_tokens.weight_int8').reshape(VOCAB, H)
emb_sc = get('model.language_model.embed_tokens.weight_scale').astype(np.float32)
lm_w8 = get('lm_head.weight_int8').reshape(VOCAB, H)
lm_sc = get('lm_head.weight_scale').astype(np.float32)

def embed(t):
    return (emb_w8[t].astype(np.float32) * emb_sc[t]).astype(np.float16).astype(np.float32)

def logits_chunked(h, topk=8):
    best_vals = np.full(topk, -np.inf, dtype=np.float64)
    best_ids = np.zeros(topk, dtype=np.int64)
    CH = 16384
    for s in range(0, VOCAB, CH):
        e = min(s + CH, VOCAB)
        W = (lm_w8[s:e].astype(np.float32) * lm_sc[s:e, None]).astype(np.float16).astype(np.float32)
        sc = W @ h
        k = min(topk, sc.shape[0])
        idx = np.argpartition(-sc, k - 1)[:k]
        vals = sc[idx]
        cat_v = np.concatenate([best_vals, vals]); cat_i = np.concatenate([best_ids, idx + s])
        sel = np.argsort(-cat_v)[:topk]
        best_vals, best_ids = cat_v[sel], cat_i[sel]
    return best_ids, best_vals

inv_freq = THETA ** (-np.arange(0, NROT, 2, dtype=np.float32) / NROT)

def rope(x, pos):
    # x [..., 256], rotate first 64 dims, interleaved pairs
    xf = x.copy()
    ang = pos * inv_freq
    cos, sin = np.cos(ang), np.sin(ang)
    for i in range(NROT // 2):
        a, b = 2 * i, 2 * i + 1
        xa, xb = xf[..., a].copy(), xf[..., b].copy()
        xf[..., a] = xa * cos[i] - xb * sin[i]
        xf[..., b] = xa * sin[i] + xb * cos[i]
    return xf

def attn_layer(li, h, t, kv):
    pre = f'model.language_model.layers.{li}.self_attn'
    x = rms(h, getn(f'model.language_model.layers.{li}.input_layernorm.weight'))
    acc('attn_norm', x)
    h0 = h
    qg = escha_mm(pre + '.q_proj', x).reshape(HEADS, 2 * HDIM)
    acc('Qcur_full', qg)
    q, gate = qg[:, :HDIM], qg[:, HDIM:]
    k = escha_mm(pre + '.k_proj', x).reshape(KVH, HDIM)
    v = escha_mm(pre + '.v_proj', x).reshape(KVH, HDIM)
    q = rms(q, getn(pre + '.q_norm.weight'))
    k = rms(k, getn(pre + '.k_norm.weight'))
    acc('Qcur_normed', q); acc('Kcur_normed', k)
    qr, kr = rope(q, t), rope(k, t)
    acc('Qcur', qr); acc('Kcur', kr); acc('Vcur', v)
    q, k = qr, kr
    kv.append((k.astype(np.float32), v.astype(np.float32)))
    ks = np.stack([kk[0] for kk in kv]); vs = np.stack([vv[1] for vv in kv])  # [t+1,KVH,256]
    rep = HEADS // KVH
    att = np.einsum('hd,shd->hs', q, np.repeat(ks, rep, axis=1)) / math.sqrt(HDIM)
    mask = np.full(att.shape[-1], -np.inf); mask[:t + 1] = 0
    att = att + mask
    att = att - att.max(axis=-1, keepdims=True)
    p = np.exp(att); p /= p.sum(axis=-1, keepdims=True)
    o = np.einsum('hs,shd->hd', p, np.repeat(vs, rep, axis=1))
    acc('attn_pregate', o)
    og = o * (1 / (1 + np.exp(-gate)))
    acc('attn_gated', og)
    return h0 + escha_mm(pre + '.o_proj', og.reshape(-1))

def mlp(li, h, pre_normed=False):
    x = h if pre_normed else rms(h, getn(f'model.language_model.layers.{li}.post_attention_layernorm.weight'))
    g = escha_mm(f'model.language_model.layers.{li}.mlp.gate_proj', x)
    u = escha_mm(f'model.language_model.layers.{li}.mlp.up_proj', x)
    act = g / (1 + np.exp(-g)) * u
    acc('ffn_gate_escha', g/(1+np.exp(-g))); acc('ffn_up_escha', u); acc('ffn_silu_par_escha', act)
    o = h + escha_mm(f'model.language_model.layers.{li}.mlp.down_proj', act)
    return o

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'
    ids = bpe_encode(prompt)
    print('ids', ids)
    h_hist = []
    kv = [[] for _ in range(NL)]
    conv_states = [dict() for _ in range(NL)]
    ssm_states = [None] * NL
    for t, tid in enumerate(ids):
        h = embed(tid)
        for li in range(NL):
            CUR_LAYER[0] = li
            if li % 4 == 3:
                ao = attn_layer(li, h, t, kv[li])
                acc('attn_output', ao)
                res = h + ao
            else:
                pre = f'model.language_model.layers.{li}.linear_attn'
                x = rms(h, getn(f'model.language_model.layers.{li}.input_layernorm.weight'))
                acc('attn_norm', x)
                mixed = escha_mm(pre + '.in_proj_qkv', x)
                acc('linear_attn_qkv_mixed', mixed)
                z = escha_mm(pre + '.in_proj_z', x).reshape(NV, DV)
                import sys as _s
                bb = get(pre + '.in_proj_b.weight').astype(np.float32) @ x
                aa = get(pre + '.in_proj_a.weight').astype(np.float32) @ x
                cw = get(pre + '.conv1d.weight').astype(np.float32).reshape(-1, CONV_K)
                if t == 0:
                    cs = np.zeros((CONV_K - 1, cw.shape[0]), dtype=np.float32)
                else:
                    cs = conv_states[li]
                buf = np.concatenate([cs, mixed[None, :]], axis=0)
                cout = np.einsum('ck,kc->c', cw, buf)
                acc('conv_output_raw', cout)
                act = cout / (1 + np.exp(-cout))
                acc('conv_output_silu', act)
                conv_states[li] = buf[1:]
                q = act[:KDIM].reshape(NK, DK); k2 = act[KDIM:2 * KDIM].reshape(NK, DK); v2 = act[2 * KDIM:].reshape(NV, DV)
                # delta-net input norm: q = 128^-1 * rmsnorm(q), k = 128^-0.5 * rmsnorm(k)
                inv = DK ** -0.5
                q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6) * (inv * inv)
                k2 = k2 / np.sqrt(np.mean(k2 * k2, axis=-1, keepdims=True) + 1e-6) * inv
                acc('q_conv_predelta', q); acc('k_conv_predelta', k2); acc('v_conv_predelta', v2)
                A_log = get(pre + '.A_log').astype(np.float32); dtb = get(pre + '.dt_bias').astype(np.float32)
                g = -np.exp(A_log) * softplus(aa + dtb); beta = 1 / (1 + np.exp(-bb))
                acc('gate', g); acc('beta_sigmoid', beta)
                if ssm_states[li] is None:
                    ssm_states[li] = [np.zeros((DK, DV), dtype=np.float32) for _ in range(NV)]
                y = np.zeros((NV, DV), dtype=np.float32)
                for j in range(NV):
                    kj = j // (NV // NK)
                    S = ssm_states[li][j]
                    S *= math.exp(g[j])
                    S += beta[j] * np.outer(k2[kj], v2[j] - S.T @ k2[kj])
                    y[j] = S.T @ q[kj]
                acc('attn_output', y)
                nw = get(pre + '.norm.weight').astype(np.float32)
                yn = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + 1e-6) * nw
                gated = yn * (z / (1 + np.exp(-z)))
                ao = escha_mm(f'model.language_model.layers.{li}.linear_attn.out_proj', gated.reshape(-1))
                acc('linear_attn_out', ao)
                res = h + ao
                CUR_TAIL[0]=False
            if CUR_TAIL[0]:
                pass
            acc('attn_residual', res)
            pn = rms(res, getn(f'model.language_model.layers.{li}.post_attention_layernorm.weight'))
            acc('attn_post_norm', pn)
            h = mlp(li, pn, pre_normed=True)
            acc('l_out', h)
            if t == len(ids) - 1:
                print(f'L{li:02d} |h|={np.linalg.norm(h):.4f}')
        h_hist.append(h)
    hn = rms(h, getn('model.language_model.norm.weight'))
    acc('h_nextn', hn)
    ids8, vals8 = logits_chunked(hn)
    acc('result_output', (lm_w8[ids8].astype(np.float32)*lm_sc[ids8,None]).astype(np.float16).astype(np.float32) @ hn)
    if _os.environ.get('NPY_DEBUG_SUMS'):
        _json.dump(SUMS, open('/tmp/npy_sums.json','w'))
    pieces = {v: k for k, v in vocab.items()}
    print('top8:', [(int(i), repr(pieces.get(int(i), '?')), round(float(v), 3)) for i, v in zip(ids8, vals8)])

if __name__ == '__main__':
    main()
