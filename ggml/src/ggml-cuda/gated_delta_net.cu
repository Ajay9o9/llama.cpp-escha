#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// Warps per block, and the floor on blocks per SM that picks C_W (columns of the
// recurrent state per warp) in launch_gated_delta_net_tokens.
// k_reg/q_reg depend only on the row index, so one load of q_t/k_t serves every
// column a warp owns: q and k are read S_v/C_W times per token per head instead of
// S_v times. Note that factor is independent of GDN_TOKEN_WARPS - warps per block
// sets the block count, C_W sets the traffic, and they trade off separately.
// That redundancy, not arithmetic, is what this kernel was bound by.
constexpr int GDN_TOKEN_WARPS = 2;
constexpr int GDN_TOKEN_MIN_WARPS_PER_SM = 2;

template <int S_v, bool KDA, bool keep_rs_t, int C_W>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * GDN_TOKEN_WARPS, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns C_W columns, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col0     = (int) (blockIdx.z * blockDim.y + threadIdx.y) * C_W;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col0 * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(C_W >= 1 && C_W <= warp_size, "C_W must fit in a warp for the output store");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[C_W][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < C_W; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i     = r * warp_size + lane;
            s_shard[c][r]   = curr_state[c * S_v + i];
        }
    }

    // Software pipeline: token t's q/k/v/g/beta are fetched while token t-1 is still
    // being reduced. Without this the global load sits at the head of a chain that
    // also holds two dependent warp reductions, and at C_W=4 that latency - not
    // bandwidth - is what the kernel is bound by.
    constexpr int g_regs = KDA ? rows_per_lane : 1;

    float k_cur[rows_per_lane], q_cur[rows_per_lane];
    float k_nxt[rows_per_lane], q_nxt[rows_per_lane];
    float v_cur[C_W],           v_nxt[C_W];
    float g_cur[g_regs],        g_nxt[g_regs];
    float b_cur = 0.0f,         b_nxt = 0.0f;

#define GDN_FETCH_TOKEN(t, kr, qr, vr, gr, br)                                            \
    do {                                                                                  \
        const float * q_t_ = q + iq3 * sq3 + (t) * sq2 + iq1 * sq1;                       \
        const float * k_t_ = k + iq3 * sq3 + (t) * sq2 + iq1 * sq1;                       \
        const float * v_t_ = v + sequence * sv3 + (t) * sv2 + h_idx * sv1;                \
        const int64_t gb_  = sequence * sb3 + (t) * sb2 + h_idx * sb1;                    \
        _Pragma("unroll")                                                                 \
        for (int r = 0; r < rows_per_lane; r++) {                                         \
            const int i_ = r * warp_size + lane;                                          \
            kr[r] = k_t_[i_];                                                             \
            qr[r] = q_t_[i_];                                                             \
        }                                                                                 \
        _Pragma("unroll")                                                                 \
        for (int c = 0; c < C_W; c++) {                                                   \
            vr[c] = v_t_[col0 + c];                                                       \
        }                                                                                 \
        if constexpr (KDA) {                                                              \
            const float * g_t_ = g + gb_ * S_v;                                           \
            _Pragma("unroll")                                                             \
            for (int r = 0; r < rows_per_lane; r++) {                                     \
                gr[r] = g_t_[r * warp_size + lane];                                       \
            }                                                                             \
        } else {                                                                          \
            gr[0] = g[gb_];                                                               \
        }                                                                                 \
        br = beta[gb_];                                                                   \
    } while (0)

    GDN_FETCH_TOKEN(0, k_cur, q_cur, v_cur, g_cur, b_cur);

    for (int t = 0; t < n_tokens; t++) {
        if (t + 1 < n_tokens) {
            GDN_FETCH_TOKEN(t + 1, k_nxt, q_nxt, v_nxt, g_nxt, b_nxt);
        }

        const float beta_val = b_cur;
        float attn_col[C_W];

        if constexpr (!KDA) {
            const float g_val = expf(g_cur[0]);

            float delta_col[C_W];
#pragma unroll
            for (int c = 0; c < C_W; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_cur[r];
                }
                const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                delta_col[c] = (v_cur[c] - g_val * kv_col) * beta_val;
            }

#pragma unroll
            for (int c = 0; c < C_W; c++) {
                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_cur[r] * delta_col[c];
                    attn_partial += s_shard[c][r] * q_cur[r];
                }
                attn_col[c] = warp_reduce_sum<warp_size>(attn_partial);
            }
        } else {
            // g is per-row here; exp once, not once per use
            float ge[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                ge[r] = expf(g_cur[r]);
            }

            float delta_col[C_W];
#pragma unroll
            for (int c = 0; c < C_W; c++) {
                // kv[col] = sum_i g[i] * S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += ge[r] * s_shard[c][r] * k_cur[r];
                }
                const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - kv[col]) * beta
                delta_col[c] = (v_cur[c] - kv_col) * beta_val;
            }

#pragma unroll
            for (int c = 0; c < C_W; c++) {
                // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = ge[r] * s_shard[c][r] + k_cur[r] * delta_col[c];
                    attn_partial += s_shard[c][r] * q_cur[r];
                }
                attn_col[c] = warp_reduce_sum<warp_size>(attn_partial);
            }
        }

        // lane c stores column c: C_W consecutive floats, one coalesced transaction
#pragma unroll
        for (int c = 0; c < C_W; c++) {
            if (lane == c) {
                attn_data[col0 + c] = attn_col[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snap_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < C_W; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        snap_state[(col0 + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_cur[r] = k_nxt[r];
            q_cur[r] = q_nxt[r];
        }
#pragma unroll
        for (int c = 0; c < C_W; c++) {
            v_cur[c] = v_nxt[c];
        }
#pragma unroll
        for (int r = 0; r < g_regs; r++) {
            g_cur[r] = g_nxt[r];
        }
        b_cur = b_nxt;
    }
#undef GDN_FETCH_TOKEN

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C_W; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i                  = r * warp_size + lane;
                state[(col0 + c) * S_v + i]  = s_shard[c][r];
            }
        }
    }
}


// GGML_CUDA_GDN_COLS: force the token-loop's columns-per-warp (1/2/4/8) instead of
// picking it from the grid size. For retuning on a GPU with a different SM count.
static int gdn_token_cols_override() {
    static const int cols = []() {
        const char * env = getenv("GGML_CUDA_GDN_COLS");
        return env != nullptr ? std::atoi(env) : 0;
    }();
    return cols;
}

template <int S_v, bool KDA, bool keep_rs_t, int C_W>
static void launch_gated_delta_net_tokens_cw(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K, int warp_size, cudaStream_t stream) {
    static_assert(S_v % (GDN_TOKEN_WARPS * C_W) == 0, "columns per block must divide S_v");

    const dim3 grid_dims(H, n_seqs, S_v / (GDN_TOKEN_WARPS * C_W));
    const dim3 block_dims(warp_size <= S_v ? warp_size : S_v, GDN_TOKEN_WARPS, 1);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t, C_W>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}

// C_W is capped at 4 from above by registers - wider spills s_shard to local memory
// (C_W=8 measured 2x slower, C_W=16 7x) - and from below by the fact that it buys
// both the q/k reuse and the ILP that hides the prefetch. What limits it is warps in
// flight: a warp owns C_W columns, so the kernel runs H*n_seqs*S_v/C_W warps
// regardless of how they are grouped into blocks. Take the widest C_W that still
// keeps GDN_TOKEN_MIN_WARPS_PER_SM resident.
template <int S_v, bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_tokens(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K,
        int warp_size, int nsm, cudaStream_t stream) {
    constexpr int MAXC = S_v / GDN_TOKEN_WARPS >= 1 ? S_v / GDN_TOKEN_WARPS : 1;
    constexpr int C8   = MAXC < 8 ? MAXC : 8;
    constexpr int C4   = MAXC < 4 ? MAXC : 4;
    constexpr int C2   = MAXC < 2 ? MAXC : 2;

    const int64_t warps_at_1 = H * n_seqs * S_v;
    const int64_t min_warps   = (int64_t) GDN_TOKEN_MIN_WARPS_PER_SM * nsm;
    const int     forced      = gdn_token_cols_override();

#define GDN_LAUNCH_TOKENS_CW(CW)                                                             \
    launch_gated_delta_net_tokens_cw<S_v, KDA, keep_rs_t, CW>(                               \
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,                   \
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,                                         \
        neqk1_magic, rq3_magic, scale, state_slot_stride, K, warp_size, stream)

    if (forced == 0) {
        if (warps_at_1 / C4 >= min_warps) {
            GDN_LAUNCH_TOKENS_CW(C4);
        } else if (warps_at_1 / C2 >= min_warps) {
            GDN_LAUNCH_TOKENS_CW(C2);
        } else {
            GDN_LAUNCH_TOKENS_CW(1);
        }
    } else if (forced >= 8) {
        GDN_LAUNCH_TOKENS_CW(C8);
    } else if (forced >= 4) {
        GDN_LAUNCH_TOKENS_CW(C4);
    } else if (forced >= 2) {
        GDN_LAUNCH_TOKENS_CW(C2);
    } else {
        GDN_LAUNCH_TOKENS_CW(1);
    }
#undef GDN_LAUNCH_TOKENS_CW
}


template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill. A chunked FLA/WY
    //      implementation was tried and measured slower than this token loop
    //      on Ampere at Qwen3.8 sizes; it needs tensor cores to pay off.
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            launch_gated_delta_net_tokens<16, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, warp_size, nsm, stream);
            break;
        case 32:
            launch_gated_delta_net_tokens<32, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, warp_size, nsm, stream);
            break;
        case 64:
            launch_gated_delta_net_tokens<64, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, warp_size, nsm, stream);
            break;
        case 128:
            launch_gated_delta_net_tokens<128, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, warp_size, nsm, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
