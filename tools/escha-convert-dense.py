#!/usr/bin/env python3
"""Convert an EschaLabs dense W2 checkpoint (qwen3_5_text) to a llama.cpp GGUF.

Keeps the escha-coded projections byte-for-byte in their native 2-bit format
(escha_code sidecars); non-coded tensors are written as F16/F32. Requires a
llama.cpp-escha build with GGML_OP_ESCHA_LINEAR (dense support).
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import gguf  # noqa: E402


class Shards:
    def __init__(self, index_file: Path):
        idx = json.load(open(index_file))
        self.weight_map = idx["weight_map"]
        self.base = index_file.parent
        self._handles = {}
        self._headers = {}

    def _load_header(self, shard: str):
        if shard not in self._headers:
            f = open(self.base / shard, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            self._handles[shard] = f
            self._headers[shard] = (hdr, 8 + n)
        return self._headers[shard]

    def get(self, name: str) -> np.ndarray:
        shard = self.weight_map[name]
        hdr, base = self._load_header(shard)
        v = hdr[name]
        f = self._handles[shard]
        f.seek(base + v["data_offsets"][0])
        nbytes = v["data_offsets"][1] - v["data_offsets"][0]
        dt = {"I32": np.int32, "F32": np.float32, "F16": np.float16,
              "I16": np.int16, "BF16": np.uint16, "I8": np.int8}[v["dtype"]]
        return np.frombuffer(f.read(nbytes), dtype=dt)

    def has(self, name: str) -> bool:
        return name in self.weight_map


def dequant_int8(w8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # w8 [OC, IC] row scales -> fp16 weights
    return (w8.astype(np.float32) * scale.astype(np.float32)[:, None]).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    args = ap.parse_args()

    cfg_full = json.load(open(args.inp / "config.json"))
    cfg = cfg_full.get("text_config", cfg_full)

    sh = Shards(args.inp / "model.safetensors.index.json")
    prefix = "model.language_model."
    n_layer = cfg["num_hidden_layers"]
    n_embd = cfg["hidden_size"]
    head_dim = cfg["head_dim"]
    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    interval = cfg["full_attention_interval"]
    dt_rank = cfg["linear_num_value_heads"]
    key_heads = cfg["linear_num_key_heads"]
    khd = cfg["linear_key_head_dim"]
    conv = cfg["linear_conv_kernel_dim"]
    key_dim = key_heads * khd
    value_dim = dt_rank * cfg["linear_value_head_dim"]
    n_ff = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    eps = cfg["rms_norm_eps"]
    theta = cfg.get("rope_parameters", {}).get("rope_theta",
            cfg.get("rope_theta", 10000000))

    writer = gguf.GGUFWriter(args.out, "qwen35")
    writer.add_name("Qwen3.8-27B-Escha-W2")
    writer.add_description("Escha native 2-bit export (dense), converted for llama.cpp-escha")
    writer.add_architecture()
    writer.add_uint32("qwen35.block_count", n_layer)
    writer.add_uint32("qwen35.context_length", cfg.get("max_position_embeddings", 262144))
    writer.add_uint32("qwen35.embedding_length", n_embd)
    writer.add_uint32("qwen35.feed_forward_length", n_ff)
    writer.add_uint32("qwen35.attention.head_count", n_head)
    writer.add_uint32("qwen35.attention.head_count_kv", n_head_kv)
    writer.add_uint32("qwen35.attention.key_length", head_dim)
    writer.add_uint32("qwen35.attention.value_length", head_dim)
    writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", eps)
    writer.add_float32("qwen35.rope.freq_base", float(theta))
    writer.add_uint32("qwen35.rope.dimension_count", int(head_dim * 0.25))
    writer.add_array("qwen35.rope.dimension_sections", [11, 11, 10, 0])
    writer.add_uint32("qwen35.ssm.conv_kernel", conv)
    writer.add_uint32("qwen35.ssm.state_size", cfg["linear_key_head_dim"])
    writer.add_uint32("qwen35.ssm.group_count", key_heads)
    writer.add_uint32("qwen35.ssm.time_step_rank", dt_rank)
    writer.add_uint32("qwen35.ssm.inner_size", value_dim)
    writer.add_uint32("qwen35.full_attention_interval", interval)
    writer.add_uint32("qwen35.escha.version", 1)

    def add_tensor(name, arr):
        ttype = gguf.GGMLQuantizationType.F32 if arr.dtype == np.float32 else (
                gguf.GGMLQuantizationType.F16 if arr.dtype == np.float16 else
                gguf.GGMLQuantizationType.I16)
        writer.add_tensor(name, arr, raw_dtype=ttype)

    def coded(dst_base: str, src: str):
        """copy one escha-coded linear's sidecars"""
        code = sh.get(prefix + src + ".escha_code")
        cfgv = sh.get(prefix + src + ".escha_config").tolist()
        ic_v, oc_v = cfgv[4], cfgv[5]
        n_code = code.size // ((ic_v // 16) * (oc_v // 16))
        # flat is (nit, nct, n_code); gguf reverses to ne [n_code, OC/16, IC/16]
        code = code.reshape(ic_v // 16, oc_v // 16, n_code)
        add_tensor(f"{dst_base}.escha_code", code)
        add_tensor(f"{dst_base}.escha_rin",  sh.get(prefix + src + ".escha_rin"))
        add_tensor(f"{dst_base}.escha_rout", sh.get(prefix + src + ".escha_rout"))
        add_tensor(f"{dst_base}.escha_s_in",  sh.get(prefix + src + ".escha_s_in"))
        add_tensor(f"{dst_base}.escha_s_out", sh.get(prefix + src + ".escha_s_out"))
        bias = sh.get(prefix + src + ".bias")
        if bias.dtype != np.float16:
            bias = bias.astype(np.float16)
        add_tensor(f"{dst_base}.escha_bias", bias)

    # embeddings / head (int8 per-row scales -> F16)
    print("embeddings...")
    emb = dequant_int8(sh.get(prefix + "embed_tokens.weight_int8").reshape(vocab, n_embd),
                       sh.get(prefix + "embed_tokens.weight_scale"))
    add_tensor("token_embd.weight", emb.astype(np.float16))           # written (vocab, n_embd) -> ne [n_embd, vocab]
    out_w = dequant_int8(sh.get("lm_head.weight_int8").reshape(vocab, n_embd),
                         sh.get("lm_head.weight_scale"))
    add_tensor("output.weight", out_w.astype(np.float16))
    add_tensor("output_norm.weight", sh.get(prefix + "norm.weight").astype(np.float32) + 1.0)  # stored as (w-1)

    print("layers...")
    for i in range(n_layer):
        lp = f"{prefix}layers.{i}"
        add_tensor(f"blk.{i}.attn_norm.weight",      sh.get(lp + ".input_layernorm.weight").astype(np.float32) + 1.0)  # stored as (w-1)
        add_tensor(f"blk.{i}.post_attention_norm.weight", sh.get(lp + ".post_attention_layernorm.weight").astype(np.float32) + 1.0)  # stored as (w-1)

        is_attn = sh.has(lp + ".self_attn.q_proj.escha_code")
        if is_attn:
            coded(f"blk.{i}.attn_q", f"layers.{i}.self_attn.q_proj")
            coded(f"blk.{i}.attn_k", f"layers.{i}.self_attn.k_proj")
            coded(f"blk.{i}.attn_v", f"layers.{i}.self_attn.v_proj")
            coded(f"blk.{i}.attn_output", f"layers.{i}.self_attn.o_proj")
            add_tensor(f"blk.{i}.attn_q_norm.weight", sh.get(lp + ".self_attn.q_norm.weight").astype(np.float32) + 1.0)  # stored as (w-1)
            add_tensor(f"blk.{i}.attn_k_norm.weight", sh.get(lp + ".self_attn.k_norm.weight").astype(np.float32) + 1.0)  # stored as (w-1)
        else:
            coded(f"blk.{i}.attn_qkv",  f"layers.{i}.linear_attn.in_proj_qkv")
            coded(f"blk.{i}.attn_gate", f"layers.{i}.linear_attn.in_proj_z")
            coded(f"blk.{i}.ssm_out",   f"layers.{i}.linear_attn.out_proj")
            conv_w = sh.get(lp + ".linear_attn.conv1d.weight")
            if conv_w.ndim != 2:
                conv_w = conv_w.reshape(value_dim + 2*key_dim, conv)
            add_tensor(f"blk.{i}.ssm_conv1d.weight", conv_w.astype(np.float32))
            add_tensor(f"blk.{i}.ssm_dt.bias",       sh.get(lp + ".linear_attn.dt_bias").astype(np.float32))
            # llama.cpp multiplies gate by ssm_a directly: store -exp(A_log)
            a_log = sh.get(lp + ".linear_attn.A_log").astype(np.float32)
            add_tensor(f"blk.{i}.ssm_a", (-np.exp(a_log)).astype(np.float32))
            add_tensor(f"blk.{i}.ssm_beta.weight",  sh.get(lp + ".linear_attn.in_proj_b.weight").reshape(dt_rank, n_embd))
            add_tensor(f"blk.{i}.ssm_alpha.weight", sh.get(lp + ".linear_attn.in_proj_a.weight").reshape(dt_rank, n_embd))
            add_tensor(f"blk.{i}.ssm_norm.weight",  sh.get(lp + ".linear_attn.norm.weight").astype(np.float32))

        coded(f"blk.{i}.ffn_gate", f"layers.{i}.mlp.gate_proj")
        coded(f"blk.{i}.ffn_up",   f"layers.{i}.mlp.up_proj")
        coded(f"blk.{i}.ffn_down", f"layers.{i}.mlp.down_proj")
        if i % 8 == 0:
            print(f"  layer {i}/{n_layer}")

    # tokenizer from tokenizer.json (BPE, gpt2-style)
    print("tokenizer...")
    tok = json.load(open(args.inp / "tokenizer.json"))["model"]
    vocab_t = tok["vocab"]          # token -> rank
    merges = tok["merges"]
    n_bpe = len(vocab_t)
    tokens = [None] * n_bpe
    scores = np.zeros(n_bpe, dtype=np.float32)
    types = np.ones(n_bpe, dtype=np.int32)   # NORMAL
    for tok_s, rank in vocab_t.items():
        tokens[rank] = tok_s
        scores[rank] = -float(rank)
    # added tokens occupy ids n_bpe .. ; specials become CONTROL
    tc = json.load(open(args.inp / "tokenizer_config.json"))
    st = tc.get("added_tokens_decoder", {})
    added = sorted(((int(k), v) for k, v in st.items()))
    for tid, info in added:
        while len(tokens) < tid:
            tokens.append(f"<dummy_{len(tokens)}>")
            scores = np.append(scores, -1e9)
            types = np.append(types, 5)      # UNUSED padding
        content_s = info.get("content", "")
        if info.get("special", False) or (content_s.startswith('<|') and content_s.endswith('|>')):
            t_type = 3                        # CONTROL
        else:
            t_type = 4                        # USER_DEFINED
        tokens.append(info["content"])
        scores = np.append(scores, -1e9)
        types = np.append(types, t_type)
    while len(tokens) < vocab:
        tokens.append(f"<dummy_{len(tokens)}>")
        scores = np.append(scores, -1e9)
        types = np.append(types, 5)          # UNUSED padding
    writer.add_uint32("qwen35.vocab_length", vocab)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores.tolist())
    writer.add_token_types(types.tolist())
    writer.add_token_merges(merges)
    writer.add_bos_token_id(cfg.get("bos_token_id") or 248044)
    writer.add_eos_token_id(cfg.get("eos_token_id") or 248044)
    tmpl = args.inp / "chat_template.jinja"
    if tmpl.exists():
        writer.add_chat_template(tmpl.read_text())
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print("done:", args.out)


if __name__ == "__main__":
    main()
