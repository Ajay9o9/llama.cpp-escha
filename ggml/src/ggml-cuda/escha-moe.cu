#include "common.cuh"
#include "escha-moe.cuh"
#include "mmid.cuh"
#include "mma.cuh"
#include <cuda_pipeline.h>

// Fused decode + routed matmul for Escha ESCHAM experts.
//
//   y = T128(T128(x * rin) @ decode(code)) * rout
//
// Weights decode independently -- weight[p] = lut[sum_j bit(payload, dep[p][j]) << j],
// no trellis state -- so the decode is a plain gather and the whole thing is one big
// embarrassingly parallel reduction.
//
// Split into three kernels because at batch 1 there is very little natural
// parallelism: 8 slots x OC/128 column groups is 32 blocks for gate/up, which leaves
// most of the GPU idle. Slicing the IC reduction across blocks and summing the
// partials afterwards is what fills it. The partials are summed in a fixed order
// rather than with atomics so results stay bit-reproducible run to run.
//
// There are two matmul kernels, picked by batch size:
//
//   escha_matmul_partial  one row per block, reduction sliced. Right when rows are
//                         scarce, but every row decodes the weights again.
//   escha_matmul_tiled    ESCHA_ROWS rows of ONE expert per block, so a decoded
//                         weight is reused across all of them. Needs the rows
//                         grouped by expert first, which is what mm_ids_helper
//                         already does for mul_mat_id.
//
// Both write the same partial layout and share escha_finalize.

#define ESCHA_TILE     16   // decode tile is 16x16
#define ESCHA_NT      128   // threads per block, one per output column
#define ESCHA_GROUPS  (ESCHA_NT/ESCHA_TILE)
#define ESCHA_MAX_W    24   // uint32 words per payload, 48 int16 at K=3
#define ESCHA_TARGET  512   // block count we try to reach by slicing the reduction
#define ESCHA_ROWS     16   // rows of one expert per block on the batched path
// dense reuses every decoded weight across this many rows. the decode, not bandwidth, is
// the limit (see the routed path's ablation), so this is the main prefill lever: 16 -> 64
// took a perplexity pass from 47.2 s to 18.9 s. kept separate from ESCHA_ROWS so the routed
// path stays exactly as measured.
//
// But acc[R] is per thread whatever the real row count is, so a big R costs occupancy at
// batch 1 and cripples generation (R=64 measured 1.84 t/s). Generation gets its own small
// instantiation instead -- the decode work per token is the same either way, so all that
// matters there is filling the device.
#define ESCHA_ROWS_DENSE      64
#define ESCHA_ROWS_DENSE_GEN   1
#define ESCHA_GEN_MAX_ROWS    16   // at or below this, use the generation instantiation
#define ESCHA_GEN_TARGET_MUL   4
// ...and at or above this many rows per block, generation stops using the
// one-thread-per-column kernel and switches to the register-tiled one. MEASURED, not
// assumed: the tile trades instructions for warps, and the trade only pays at the top of the
// range. R=16 gained 9% (87.3 -> 95.2 tok/s at batch 16); R=8 LOST 31% (77.7 -> 53.9 at
// batch 8), because there a decoded weight feeds 4 MACs in the owning thread instead of 8,
// and it is the per-THREAD amortisation that sets the instruction count. See the launch
// site for the full accounting.
#define ESCHA_GEN_TILE_MIN    16
// ...but "generation" was only ever one row per block, so a 2-16 row batch decoded the
// whole weight payload once PER ROW. The decode is ~12x more expensive than the global
// traffic it feeds (batch 1 runs at ~1/12 of this card's bandwidth), so that made the
// window that speculative decoding lands in scale linearly with rows: batch 16 was 1.31x
// batch 1 where a healthy kernel gets 6.8x.
// The R != 1 body of escha_matmul_dense already holds acc[R] live across one decoded
// weight, so all this needs is a row tile that actually covers the batch. R is rounded up
// to a power of two so it stays a template constant, and capped at ESCHA_GEN_MAX_ROWS
// (acc[R] is per thread; R=64 was measured at 1.84 t/s for exactly that reason).
// R == 1 keeps the old whole-slice u staging and the old grid, bit for bit, so
// single-row generation is untouched.
static inline int escha_gen_rows(int n_rows) {
    int r = 1;
    while (r < n_rows && r < ESCHA_GEN_MAX_ROWS) {
        r *= 2;
    }
    return r;
}
// prefill tile for the register-tiled kernel. BM*BN = TM*TN*NT, so these four fix the
// thread count too: NT = (BM/TM)*(BN/TN). BN stays 128 -- activation traffic is
// rows*IC*(OC/BN), so a narrower BN buys reuse with global bandwidth, which is a losing trade.
#define ESCHA_BM 128
#define ESCHA_BN 128
#define ESCHA_MMA_BM 128   // tensor-core prefill tile. Accumulators per thread are
#define ESCHA_MMA_BN 128   //   BM*BN/256, so BM drives register pressure and occupancy.
#define ESCHA_TM   8
#define ESCHA_TN   8
                                   //     prefill: at batch 1 there are only n_ocb blocks
                                   //     before slicing (136 for the FFN), which leaves an
                                   //     82-SM device idle. Swept 1/2/4/8/16/32 -- flat from
                                   //     10.3 to 11.3 tok/s, 4 marginally best.

// Escha codebook A, the one this checkpoint uses (its config leaves "codebook" unset,
// which eschamoe.py defaults to cbA / codebook_id 1). It is computed, not stored -- the
// same QTIP-family trick as 3INST but with its own multiplier and no addend. Recovered
// from escham_reconstruct_kernel<1, K> and checked against all 65536 entries.
// The fp16 add must stay in fp16 to match the table bit for bit.
static __device__ __forceinline__ float escha_codebook(uint32_t idx) {
    // (x & 0x8fff8fff) ^ 0x3b603b60 is one 3-input logic op; spelling it as lop3 stops
    // ptxas from splitting it across the two 16-bit lanes. immLut 0x6a = (a & b) ^ c.
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __half2float(__hadd(__low2half(h), __high2half(h)));
}

// as escha_codebook, but stopping at the half. The codebook's last operation is already
// an fp16 add, so the fp16 weight is exact -- only the ACTIVATIONS lose precision below.
static __device__ __forceinline__ half escha_codebook_h(uint32_t idx) {
    uint32_t x = idx*0xcbac1fedu;
    asm("lop3.b32 %0, %1, %2, %3, 0x6a;"
        : "=r"(x) : "r"(x), "n"(0x8fff8fffu), "n"(0x3b603b60u));
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __hadd(__low2half(h), __high2half(h));
}

// in-place normalized Sylvester-Hadamard over each block of 128
static __device__ __forceinline__ void escha_hadamard_128(float * v, int n, int tid, int nt) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n/128)*64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len)*(2*len) + (j % len);

            float * b = v + blk*128 + i;
            const float a0 = b[0];
            const float a1 = b[len];

            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }

    const float scale = rsqrtf(128.0f);
    for (int i = tid; i < n; i += nt) {
        v[i] *= scale;
    }
    __syncthreads();
}

// u[row] = T128(x[row] * rin[expert]) -- hoisted out of the matmul so the column
// blocks do not each redo it
static __global__ void escha_rotate_in(
        const half    * __restrict__ rin,
        const float   * __restrict__ x,
        const int32_t * __restrict__ ids,
        float         * __restrict__ u,
        const int IC, const int n_x, const int n_ids,
        const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ float s_u[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int it  = row / n_ids;
    const int is  = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);

    const half  * rin_e = rin + (int64_t) e*IC;
    const float * x_row = (const float *)((const char *) x + (int64_t)(is % n_x)*nb_x1 + it*nb_x2);

    for (int i = tid; i < IC; i += blockDim.x) {
        s_u[i] = x_row[i]*__half2float(rin_e[i]);
    }
    __syncthreads();

    escha_hadamard_128(s_u, IC, tid, blockDim.x);

    float * dst = u + (int64_t) row*IC;
    for (int i = tid; i < IC; i += blockDim.x) {
        dst[i] = s_u[i];
    }
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K>
static __global__ void escha_matmul_partial(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1) {
    extern __shared__ char s_raw[];

    // dep is stored transposed and packed two entries per word: [b/2][r][cc]. that makes
    // the 16 threads of a group read 16 consecutive words, which is conflict-free -- the
    // natural [p][b] layout puts them 32 bytes apart and costs an 8-way bank conflict
    uint32_t * s_dep = (uint32_t *) s_raw;                 // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                      // [ESCHA_GROUPS][ESCHA_MAX_W]

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int row   = blockIdx.x;
    const int ocb   = blockIdx.y;
    const int slice = blockIdx.z;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }
    __syncthreads();

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = ocb*ESCHA_GROUPS + grp;

    const int per   = nit/n_slices;
    const int ti0   = slice*per;
    const float * u_row = u + (int64_t) row*IC;

    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;
    float sum = 0.0f;

    for (int ti = ti0; ti < ti0 + per; ++ti) {
        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int w = cc; w < n_wd; w += ESCHA_TILE) {
            pay[w] = src[w];
        }
        __syncwarp();

        const float * uu = u_row + ti*ESCHA_TILE;

        #pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            sum += uu[r]*escha_codebook(idx);
        }
        __syncwarp();
    }

    partial[((int64_t) slice*n_rows + row)*OC + ocb*ESCHA_NT + tid] = sum;
}

// one work item per block of the tiled kernel: up to ESCHA_ROWS compact rows of expert e.
// order does not matter -- items write disjoint output rows -- so an atomic counter is enough
static __global__ void escha_build_work(
        const int32_t * __restrict__ bounds,
        int4          * __restrict__ work,
        int32_t       * __restrict__ n_work,
        const int n_expert) {
    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_expert; e += gridDim.x*blockDim.x) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        for (int s = lo; s < hi; s += ESCHA_ROWS) {
            work[atomicAdd(n_work, 1)] = make_int4(e, s, min(ESCHA_ROWS, hi - s), 0);
        }
    }
}

// partial[row][c] = u[row] . decode(code), for R rows sharing one expert
template <int K, int R>
static __global__ void escha_matmul_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        const int32_t * __restrict__ ids_dst,
        const int4    * __restrict__ work,
        const int32_t * __restrict__ n_work,
        float         * __restrict__ partial,
        const int IC, const int OC) {
    extern __shared__ char s_raw[];

    uint32_t * s_dep = (uint32_t *) s_raw;                             // [8][16][16]
    uint32_t * s_pay = s_dep + 8*256;                                  // [ESCHA_GROUPS][ESCHA_MAX_W]
    float    * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);    // [R][16]

    // the grid is sized to an upper bound, so the tail blocks have nothing to do.
    // uniform across the block, so the syncs below are still safe
    if (blockIdx.x >= (unsigned) *n_work) {
        return;
    }

    const int4 w = work[blockIdx.x];
    const int e     = w.x;
    const int start = w.y;
    const int nrow  = w.z;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid = threadIdx.x;

    for (int j = tid; j < 8*256; j += ESCHA_NT) {
        const int b2 = j / 256;
        const int p  = j % 256;
        s_dep[j] = (uint32_t) (uint16_t) dep[p*16 + 2*b2]
                 | ((uint32_t) (uint16_t) dep[p*16 + 2*b2 + 1] << 16);
    }

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    const int16_t * code_e = code + (int64_t) e*nit*nct*(16*K);
    uint32_t * pay = s_pay + grp*ESCHA_MAX_W;

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    __syncthreads();

    for (int ti = 0; ti < nit; ++ti) {
        // the whole block cooperates on one 16-wide slice of u per row, then every
        // thread reads all of it -- 16 threads of a group hit the same address, so
        // the reads broadcast instead of conflicting
        for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
            const int m = j / ESCHA_TILE;
            const int r = j % ESCHA_TILE;
            s_u[j] = m < nrow ? u[(int64_t) ids_dst[start + m]*IC + ti*ESCHA_TILE + r] : 0.0f;
        }

        const uint32_t * src = (const uint32_t *)(code_e + (int64_t)(ti*nct + tj)*(16*K));
        for (int wd = cc; wd < n_wd; wd += ESCHA_TILE) {
            pay[wd] = src[wd];
        }
        __syncthreads();

#pragma unroll 4
        for (int r = 0; r < ESCHA_TILE; ++r) {
            const uint32_t * d = s_dep + r*ESCHA_TILE + cc;

            uint32_t idx = 0;
#pragma unroll
            for (int b2 = 0; b2 < 8; ++b2) {
                const uint32_t dd = d[b2*256];
                const int d0 = dd & 0xffff;
                const int d1 = dd >> 16;

                idx |= ((pay[d0 >> 5] >> (d0 & 31)) & 1u) << (2*b2);
                idx |= ((pay[d1 >> 5] >> (d1 & 31)) & 1u) << (2*b2 + 1);
            }

            const float wv = escha_codebook(idx);
#pragma unroll
            for (int m = 0; m < R; ++m) {
                acc[m] += s_u[m*ESCHA_TILE + r]*wv;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < nrow; ++m) {
        partial[(int64_t) ids_dst[start + m]*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices, rotate the 128-column group, scale by rout
static __global__ void escha_finalize(
        const half    * __restrict__ rout,
        const int32_t * __restrict__ ids,
        const float   * __restrict__ partial,
        float         * __restrict__ dst,
        const int OC, const int n_ids, const int n_rows, const int n_slices,
        const int64_t nb_i0, const int64_t nb_i1,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int ocb = blockIdx.y;

    const int it = row / n_ids;
    const int is = row % n_ids;

    const int32_t e = *(const int32_t *)((const char *) ids + is*nb_i0 + it*nb_i1);
    const int c = ocb*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + is*nb_d1 + it*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[(int64_t) e*OC + c]);
}

void ggml_cuda_op_escha_moe(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];
    const ggml_tensor * ids  = dst->src[6];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_expert = code->ne[3];
    const int n_ids    = ids->ne[0];
    const int n_tokens = ids->ne[1];
    const int n_rows   = n_ids*n_tokens;
    const int n_ocb    = OC/ESCHA_NT;

    // reuse only pays once a block can expect a decent share of ESCHA_ROWS rows for
    // its expert. below that the sliced kernel wins, and at batch 1 it is the only
    // one that fills the device at all
    const bool tiled = n_rows >= 8*n_expert;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> u_buf(ctx.pool(), (size_t) n_rows*IC);

    escha_rotate_in<<<n_rows, 256, IC*sizeof(float), stream>>>(
        (const half *) rin->data, (const float *) x->data, (const int32_t *) ids->data,
        u_buf.get(), IC, (int) x->ne[1], n_ids,
        x->nb[1], x->nb[2], ids->nb[0], ids->nb[1]);

    const size_t smem_dep = 8*256*sizeof(uint32_t) + ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint32_t);

    if (tiled) {
        // mm_ids_helper assumes a token uses an expert at most once, which top-k routing
        // guarantees. duplicates would desync its offsets and write out of bounds
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));

        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_expert + 1);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            n_expert, n_tokens, n_ids, (int) x->ne[1], (int) (ids->nb[1]/ggml_element_size(ids)), 1,
            /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        // an expert can end with a part-full chunk, so the item count is bounded but not known
        const int n_work_max = (n_rows + ESCHA_ROWS - 1)/ESCHA_ROWS + n_expert;

        ggml_cuda_pool_alloc<int4>    work(ctx.pool(), n_work_max);
        ggml_cuda_pool_alloc<int32_t> n_work(ctx.pool(), 1);

        CUDA_CHECK(cudaMemsetAsync(n_work.get(), 0, sizeof(int32_t), stream));
        escha_build_work<<<1, 256, 0, stream>>>(bounds.get(), work.get(), n_work.get(), n_expert);

        ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_rows*OC);

        const size_t smem = smem_dep + ESCHA_ROWS*ESCHA_TILE*sizeof(float);

        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_work_max, n_ocb), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                u_buf.get(), ids_dst.get(), work.get(), n_work.get(), p_buf.get(), IC, OC);
        };

        switch (K) {
            case 2: launch(escha_matmul_tiled<2, ESCHA_ROWS>); break;
            case 3: launch(escha_matmul_tiled<3, ESCHA_ROWS>); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }

        escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
            (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
            OC, n_ids, n_rows, 1, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
        return;
    }

    // slice the reduction until the launch is wide enough to fill the device, but only
    // by factors that divide nit evenly
    int n_slices = 1;
    while (n_rows*n_ocb*n_slices*2 <= ESCHA_TARGET && nit % (n_slices*2) == 0) {
        n_slices *= 2;
    }

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    const dim3 grid(n_rows, n_ocb, n_slices);

    auto launch = [&](auto kernel) {
        kernel<<<grid, ESCHA_NT, smem_dep, stream>>>(
            (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
            u_buf.get(), (const int32_t *) ids->data, p_buf.get(),
            IC, OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1]);
    };

    switch (K) {
        case 2: launch(escha_matmul_partial<2>); break;
        case 3: launch(escha_matmul_partial<3>); break;
        default: GGML_ABORT("escha: unsupported K=%d", K);
    }

    escha_finalize<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, (const int32_t *) ids->data, p_buf.get(), (float *) dst->data,
        OC, n_ids, n_rows, n_slices, ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2]);
}

// ===========================================================================
// dense escha (ggml_escha_mul_mat)
//
// Same codec and same rotations as the routed path above, minus the routing: one weight
// matrix, every row goes through it. That removes the ids indirection, mm_ids_helper and
// the work list -- a block owns R consecutive rows outright. The IC reduction is still
// sliced across blocks, because at batch 1 a single row would otherwise leave the device
// mostly idle.
// ===========================================================================

// u[row] = T128(x[row] * rin)
//
// Staged through shared memory in fixed chunks rather than all of IC at once: the dense
// projections go up to IC = 17408 (mlp.down), and 17408 floats is 68 KB, well past the
// 48 KB a block gets. The rotation is independent per 128-block, so any chunk that is a
// multiple of 128 splits it exactly.
#define ESCHA_ROT_CHUNK 2048   // 8 KB of shared memory

// U is float for the scalar paths and half for the tensor-core path. Emitting half here
// rather than converting during staging is bit-identical -- the same __float2half, moved
// earlier -- and it is what lets cp.async copy activations straight into shared, since
// cp.async moves bytes verbatim and cannot convert.
template <typename U>
static __global__ void escha_rotate_in_dense(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        U           * __restrict__ u,
        const int IC, const int ne1,
        const int64_t nb_x1, const int64_t nb_x2) {
    __shared__ float s_u[ESCHA_ROT_CHUNK];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                           + (int64_t)(row / ne1)*nb_x2);
    U * dst = u + (int64_t) row*IC;

    for (int off = 0; off < IC; off += ESCHA_ROT_CHUNK) {
        const int n = min(ESCHA_ROT_CHUNK, IC - off);

        for (int i = tid; i < n; i += blockDim.x) {
            s_u[i] = x_row[off + i]*__half2float(rin[off + i]);
        }
        __syncthreads();

        escha_hadamard_128(s_u, n, tid, blockDim.x);

        for (int i = tid; i < n; i += blockDim.x) {
            if constexpr (sizeof(U) == sizeof(half)) {
                dst[off + i] = __float2half(s_u[i]);
            } else {
                dst[off + i] = s_u[i];
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Wide rotate.
//
// The kernel above is one BLOCK per row: at batch 1 that is a single block on a 36-SM
// device walking IC = 5120..17408 in serial 2048-float chunks, each chunk a separate
// global round trip behind a __syncthreads. ncu measured 17.7% occupancy, 0.07 waves/SM
// and ~15.3 us per launch, flat in batch size -- ~400 launches a token, ~6.2 ms, 11.5%
// of decode at batch 1.
//
// But T128 is independent per 128-element block, so there is nothing serial about it:
// n_rows*(IC/128) independent 128-point transforms. This version gives each WARP one of
// them, four elements per lane, and spreads them over a real grid.
//
// The lane mapping is i = 32*t + lane (not 4*lane + t) so that every global access is a
// fully coalesced 128-byte transaction with no alignment requirement. It puts butterfly
// bits 0-4 across the lanes (shfl_xor) and bits 5-6 inside the registers.
//
// Bit for bit identical to escha_hadamard_128: for a pair (lo,hi) differing in exactly
// one bit, lo gets lo+hi and hi gets lo-hi, applied in bit order 0..6, then one rsqrt
// scale at the end. Same operands, same operations, just a different owner.
static __device__ __forceinline__ void escha_hadamard_128_warp(float * v, int lane) {
    // len = 1,2,4,8,16 -- partner is in lane ^ len
#pragma unroll
    for (int b = 0; b < 5; ++b) {
        const int  m     = 1 << b;
        const bool upper = (lane & m) != 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float o = __shfl_xor_sync(0xffffffffu, v[t], m);
            v[t] = upper ? (o - v[t]) : (v[t] + o);
        }
    }
    // len = 32 -- partner is t ^ 1
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a1; v[1] = a0 - a1;
        v[2] = a2 + a3; v[3] = a2 - a3;
    }
    // len = 64 -- partner is t ^ 2
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a2; v[2] = a0 - a2;
        v[1] = a1 + a3; v[3] = a1 - a3;
    }
    const float scale = rsqrtf(128.0f);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        v[t] *= scale;
    }
}

// u[row] = T128(x[row] * rin), one warp per 128-block, grid-strided over all of them.
template <typename U>
static __global__ void escha_rotate_in_dense_warp(
        const half  * __restrict__ rin,
        const float * __restrict__ x,
        U           * __restrict__ u,
        const int IC, const int ne1, const int hb_per_row, const int64_t n_hb,
        const int64_t nb_x1, const int64_t nb_x2) {
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int nwarps = blockDim.x  >> 5;

    for (int64_t h = (int64_t) blockIdx.x*nwarps + warp; h < n_hb;
                 h += (int64_t) gridDim.x*nwarps) {
        const int row = (int) (h / hb_per_row);
        const int off = (int) (h % hb_per_row)*128 + lane;

        const float * x_row = (const float *)((const char *) x + (int64_t)(row % ne1)*nb_x1
                                                               + (int64_t)(row / ne1)*nb_x2);
        float v[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = off + 32*t;
            v[t] = x_row[i]*__half2float(rin[i]);
        }

        escha_hadamard_128_warp(v, lane);

        U * dst = u + (int64_t) row*IC;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = off + 32*t;
            if constexpr (sizeof(U) == sizeof(half)) {
                dst[i] = __float2half(v[t]);
            } else {
                dst[i] = v[t];
            }
        }
    }
}

#define ESCHA_ROT_NT 32    // 1 warp, i.e. one 128-block, per CUDA block

// escha's dep table is computable, so the dense kernel never reads it.
//
// The 16 bits that form a weight's codebook index are always 16 cyclically-consecutive
// positions of a bit-stream over the tile payload, where the stream visits 32-bit word 0
// first and then walks the words downwards (0, NW-1, NW-2, ...). Only the start position
// varies, and it is affine in the tile row and column:
//
//   pi(r) = (r&1) | ((r>>3)&1)<<1 | ((r>>1)&3)<<3      // bit 2 is left free for c
//   t     = pi(r) + 32*c + 4*(c>>3)
//   s     = ((32-K) - K*t) mod 256K
//
// Verified exact against both shipped tables, all 4096 entries (dep3.py). This turns
// 8 shared dep reads + 16 payload reads + ~48 bit ops per weight into two reads and a
// funnel shift, and drops the 8 KB per-block dep table that made batch 1 setup-bound.
__device__ __forceinline__ int escha_dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

// Register-tiled variant of the dense kernel, for prefill.
//
// The column-per-thread kernel below couples decode reuse to one thread's registers: it
// accumulates R rows in acc[R], so reusing a decode more means more registers in the SAME
// thread. At R=64 that is 255 registers with spill, 2 blocks/SM, ~17% occupancy -- and it
// is why prefill decodes at ~60 G/s while the batch-1 path manages ~372 G/s.
//
// Here the decoded weights go to SHARED memory instead, so every row group in the block
// reuses them. Reuse becomes BM (rows per BLOCK) while registers stay TM*TN (the thread's
// own output tile), which decouples the two:
//
//     BM * BN = (TM * TN) * NT
//
// BN is held at 128 on purpose: total activation traffic is rows * IC * (OC/BN), so
// narrowing BN to buy reuse would multiply global reads instead (that mistake was caught
// on paper, not in silicon -- BN=16 would have cost 8x the activation bandwidth).
//
// TN must be > 1. With one column per thread every shared read feeds exactly one MAC and
// shared bandwidth becomes the new ceiling; a TMxTN tile does TM*TN MACs per TM+TN reads.
template <int K, int BM, int BN, int TM, int TN>
static __global__ void escha_matmul_dense_tiled(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    constexpr int NT  = (BM/TM)*(BN/TN);   // threads per block
    constexpr int NTJ = BN/ESCHA_TILE;     // output tiles covered
    constexpr int NCX = BN/TN;             // threads across the column axis

    // s_w is padded to BN+2 because the decode below writes it r-major with r = tid%16:
    // at stride BN every thread of a warp that shares a column lands on the same bank
    // (BN*4 is a multiple of the 32-bank period), a 16-way conflict. BN+2 makes the row
    // stride 2 banks, so the 16 r values fan out over 16 even banks and the two column
    // groups take the odd ones -- conflict-free, for 64 extra floats of shared.
    constexpr int SW = BN + 2;

    extern __shared__ char s_raw[];
    // the payload is held AS overlapping adjacent pairs, as escha_matmul_dense does, so the
    // funnel shift below is one aligned LDS.64 instead of two LDS.32
    uint2 * s_pay = (uint2 *) s_raw;                                // [NTJ][ESCHA_MAX_W]
    float * s_w   = (float *)(s_pay + NTJ*ESCHA_MAX_W);             // [16][SW]
    float * s_u   = s_w + ESCHA_TILE*SW;                            // [2][16][BM] transposed

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NW = 8*K;
    const int NB = 32*NW;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid  = threadIdx.x;
    const int cx   = tid % NCX;            // this thread's column strip
    const int ry   = tid / NCX;            // this thread's row strip
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    float acc[TM*TN];
#pragma unroll
    for (int i = 0; i < TM*TN; ++i) {
        acc[i] = 0.0f;
    }

    // Decode assignment, and the whole reason a 2D tile beats the column-per-thread kernel
    // at generation sizes.
    //
    // A weight's bit offset depends only on (r, column-within-tile). The column-per-thread
    // kernel fixes the column and walks r, so the offset moves every weight and it pays
    // ~21 instructions per decode to re-derive the word indices and shift. Here a thread
    // can instead hold BOTH r and the column-within-tile fixed and walk the TILE index,
    // which does not enter the offset at all: the shift and the two word indices become
    // per-thread constants, hoisted clean out of the ti loop, and a decode collapses to one
    // LDS.64, a funnel shift, the codebook and a store.
    //
    // This is the mapping escha_matmul_dense_tiled_mma already uses. It needs one thread per
    // (r, column-within-tile) pair, hence NT = 16*16.
    static_assert(NT == ESCHA_TILE*ESCHA_TILE, "escha: decode mapping needs NT = 16*16");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    constexpr int DPT = (ESCHA_TILE*BN)/NT;    // weights this thread decodes per input tile

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every tile, every step
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NW - dg0) : 0;
    const int dsh = dsp & 31;

    // this thread's payload slot, fixed for the whole loop. One word per thread only:
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // stage tile lo's activations into buffer 0 before the loop, so that inside the loop
    // the fetch for ti+1 can be issued into the OTHER buffer and overlap this tile's work
    if (lo < hi) {
        for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
            const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
            const int row = row0 + m;
            s_u[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + lo*ESCHA_TILE + r] : 0.0f;
        }
    }

    for (int ti = lo; ti < hi; ++ti) {
        float * su_cur = s_u + (((ti - lo) & 1)      )*(ESCHA_TILE*BM);
        float * su_nxt = s_u + (((ti - lo) & 1) ^ 1  )*(ESCHA_TILE*BM);
        // publish the payload fetched last round, then issue the next fetch before the
        // barrier, so the global latency overlaps the decode instead of stalling every warp
        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            s_pay[pt*ESCHA_MAX_W + pw].y = ppre;
            s_pay[pt*ESCHA_MAX_W + (pw + 1 == NW ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        __syncthreads();

        // next tile's activations go to the other buffer: no barrier separates them from
        // the reads of su_cur below, and the barrier at the top of the next iteration is
        // what makes them visible
        if (ti + 1 < hi) {
            for (int j = tid; j < BM*ESCHA_TILE; j += NT) {
                const int m = j / ESCHA_TILE, r = j % ESCHA_TILE;
                const int row = row0 + m;
                su_nxt[r*BM + m] = row < n_rows ? u[(int64_t) row*IC + (ti + 1)*ESCHA_TILE + r] : 0.0f;
            }
        }

        // decode this input tile's 16 x BN weights once, for the whole block. All the
        // addressing is precomputed above; step k reads payload tile k and writes column
        // dccl + 16k, so this is just load, shift, codebook, store.
#pragma unroll
        for (int k = 0; k < DPT; ++k) {
            const uint2 pay = s_pay[k*ESCHA_MAX_W + dw0];
            s_w[dr*SW + dccl + ESCHA_TILE*k] =
                escha_codebook(__funnelshift_r(pay.y, pay.x, dsh) & 0xffffu);
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < ESCHA_TILE; ++r) {
            float a[TM], b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                a[m] = su_cur[r*BM + ry*TM + m];
            }
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                b[n] = s_w[r*SW + cx*TN + n];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m*TN + n] += a[m]*b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = row0 + ry*TM + m;
        if (row < n_rows) {
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                partial[((int64_t) sl*n_rows + row)*OC + oc0 + cx*TN + n] = acc[m*TN + n];
            }
        }
    }
}


// Prefill on tensor cores. Same decode as escha_matmul_dense_tiled, but the GEMM runs on
// m16n8k16 HMMA with fp32 accumulate, which on GA102 is 71 TFLOPS against 35.6 for FP32 FMA.
//
// Two layout changes fall out of the fragment shapes, and both are cheap:
//   s_u becomes [m][k] (was [k][m]) -- which makes staging a straight contiguous copy
//   s_w becomes [n][k] (was [k][n]) -- B is the ".col" operand of mma.row.col
//
// The weights stay exact (escha_codebook_h). The ACTIVATIONS are rounded to fp16, which is
// what escha's own runtime does, and costs rel_rms ~2.1e-4 against the fp32 reference.
template <int K, int BM, int BN, int WMT>
static __global__ void __launch_bounds__(256, 1) escha_matmul_dense_tiled_mma(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps
    constexpr int WM   = WMT;            // warps down the row axis
    constexpr int WN   = NW/WM;          // warps across the column axis
    static_assert(WM*WN == NW,           "escha: warp grid does not cover the block");
    static_assert(BM % (16*WM) == 0,     "escha: row tile does not split over the warp rows");
    static_assert(BN % (8*WN)  == 0,     "escha: col tile does not split over the warp cols");
    constexpr int MT   = BM/16/WM;       // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/WN;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // output tiles whose payload this block holds

    extern __shared__ char s_raw[];
    uint2    * s_pay = (uint2 *) s_raw;                               // [NTJ][ESCHA_MAX_W] pairs
    half     * s_u   = (half *)(s_pay + NTJ*ESCHA_MAX_W);             // [2][BM][16]
    half     * s_w   = s_u + 2*BM*ESCHA_TILE;                         // [BN][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NWD = 8*K;
    const int NB  = 32*NWD;

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;          // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;        // flat id, for the layout-agnostic staging loops
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int wm   = warp / WN;
    const int wn   = warp % WN;

    constexpr int DPT = (ESCHA_TILE*BN)/NT;   // weights this thread decodes per tile
    static_assert(NT % ESCHA_TILE == 0,        "escha: r would not be thread-invariant");
    static_assert((ESCHA_TILE*BN) % NT == 0,   "escha: ragged decode assignment");
    static_assert(NT/ESCHA_TILE <= ESCHA_TILE, "escha: ccl would not be thread-invariant");

    // cp.async moves 16 bytes = 8 halves per thread; BM*ESCHA_TILE halves is exactly
    // NT*8 at BM=128/NT=256, so every thread issues one copy and none loops.
    constexpr int CPB = (BM*ESCHA_TILE*(int) sizeof(half))/NT;   // bytes per thread per tile
    static_assert(CPB == 4 || CPB == 8 || CPB == 16,     "escha: cp.async wants 4, 8 or 16 B");
    static_assert(BM*ESCHA_TILE*sizeof(half) == NT*CPB,  "escha: activation copy is ragged");
    constexpr int CPT = (ESCHA_TILE*(int) sizeof(half))/CPB;    // threads sharing one row
    const int cp_m  = tid / CPT;                                // row this thread copies into
    const int cp_h  = (tid % CPT)*(CPB/(int) sizeof(half));

    const int dr   = tid % ESCHA_TILE;         // this thread's r, every k, every tile
    const int dccl = tid / ESCHA_TILE;         // and its column within the 16-wide tile
    int dsp = ((32 - K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB;
    if (dsp < 0) {
        dsp += NB;
    }
    const int dg0 = dsp >> 5;
    const int dw0 = dg0 ? (NWD - dg0) : 0;
    const int dw1 = dw0 ? (dw0 - 1)   : (NWD - 1);
    const int dsh = dsp & 31;

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;

    tile_c acc[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    // activations for tile lo into buffer 0
    if (lo < hi) {
        {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
        }
        __pipeline_commit();
    }

    for (int ti = lo; ti < hi; ++ti) {
        half * su_cur = s_u + (((ti - lo) & 1)     )*(BM*ESCHA_TILE);
        half * su_nxt = s_u + (((ti - lo) & 1) ^ 1 )*(BM*ESCHA_TILE);

        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            s_pay[pt*ESCHA_MAX_W + pw].y = ppre;
            s_pay[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        // the copy for THIS tile was committed last round; drain it before the barrier
        // that publishes s_pay, so su_cur is visible to every warp below
        __pipeline_wait_prior(0);
        __syncthreads();

        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }

        // decode into [n][k]. All the addressing is precomputed above; iteration k reads
        // payload tile k and writes column dccl + 16k, so this is just load, shift, codebook.
#pragma unroll
        for (int k = 0; k < DPT; ++k) {
            const uint2 * pay = s_pay + k*ESCHA_MAX_W;
            const int c = dccl + ESCHA_TILE*k;
            s_w[c*ESCHA_TILE + dr] =
                escha_codebook_h(__funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffffu);
        }
        __syncthreads();

        {
            const half2 * su2 = (const half2 *) su_cur;   // [BM][8] half2
            const half2 * sw2 = (const half2 *) s_w;      // [BN][8] half2

            tile_a A[MT];
            tile_b B[NTT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(wm*(16*MT) + i*16)*8, 8);
            }
#pragma unroll
            for (int j = 0; j < NTT; ++j) {
                ggml_cuda_mma::load_ldmatrix(B[j], sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j = 0; j < NTT; ++j) {
                    ggml_cuda_mma::mma(acc[i][j], A[i], B[j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NTT; ++j) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; ++l) {
                const int m   = wm*(16*MT) + i*16 + tile_c::get_i(l);
                const int n   = wn*(8*NTT) + j*8  + tile_c::get_j(l);
                const int row = row0 + m;
                if (row < n_rows) {
                    partial[((int64_t) sl*n_rows + row)*OC + oc0 + n] = acc[i][j].x[l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(code, lut, dep, u, partial, IC, OC, n_rows, n_slices);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// Prefill on tensor cores, decoding straight into the B fragments.
//
// escha_matmul_dense_tiled_mma stages every decoded weight through shared memory: an
// STS.U16 per weight, a block barrier, then ldmatrix to read it back -- and with WM warp
// rows each B fragment is re-read WM times. Laying all the warps across the column axis
// instead (WM = 1) gives every warp its own 16 columns, which is exactly ONE payload tile,
// so a warp can decode straight into the registers the mma wants. The shared round trip,
// its barrier and the B-side ldmatrix all disappear, and no weight is decoded twice.
//
// It also halves the payload loads. A B fragment element is a half2 holding k = 2j and
// k = 2j+1, and pi(2j+1) = pi(2j) + 1 exactly, so those two weights start K bits apart in
// the same bit stream: one 32-bit window covers both (16 + K <= 19 bits) and the second
// index is the first shifted down by K. One LDS.64 and one funnel shift now feed two
// weights instead of one.
//
// The arithmetic is untouched: same escha_codebook_h, same m16n8k16 accumulate over the
// same k, same slice boundaries. Only which thread decodes which weight changes.
template <int K, int BM, int BN, int MINB>
static __global__ void __launch_bounds__(256, MINB) escha_matmul_dense_mma_rb(
        const int16_t * __restrict__ code,
        const half    * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
#ifdef TURING_MMA_AVAILABLE
    constexpr int NT   = 256;
    constexpr int NW   = NT/32;          // warps, all of them across the column axis
    constexpr int MT   = BM/16;          // 16-row accumulator tiles per warp
    constexpr int NTT  = BN/8/NW;        // 8-col accumulator tiles per warp
    constexpr int NTJ  = BN/ESCHA_TILE;  // payload tiles this block holds
    static_assert(NTJ == NW,             "escha: rb wants exactly one payload tile per warp");
    static_assert(BM % 16 == 0,          "escha: rb row tile must be a multiple of 16");
    static_assert(NTT*8*NW == BN,        "escha: rb column tile does not divide");

    extern __shared__ char s_raw[];
    uint2 * s_pay = (uint2 *) s_raw;                              // [2][NTJ][ESCHA_MAX_W]
    half  * s_u   = (half *)(s_pay + 2*NTJ*ESCHA_MAX_W);          // [2][BM][16]

    const int NWD = 8*K;                 // 32-bit words in a tile payload
    const int NB  = 32*NWD;              // and bits

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int lane = threadIdx.x;        // must stay the lane: mma.cuh indexes on it
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;
    const int row0 = blockIdx.x*BM;
    const int oc0  = blockIdx.y*BN;

    const int sl = blockIdx.z;
    const int lo = (int) (((int64_t) nit*sl)/n_slices);
    const int hi = (int) (((int64_t) nit*(sl + 1))/n_slices);

    // cp.async moves CPB bytes per thread; BM*16 halves spread over NT threads
    constexpr int CPB = (BM*ESCHA_TILE*(int) sizeof(half))/NT;
    static_assert(CPB == 4 || CPB == 8 || CPB == 16, "escha: cp.async wants 4, 8 or 16 B");
    constexpr int CPT = (ESCHA_TILE*(int) sizeof(half))/CPB;
    const int cp_m = tid / CPT;
    const int cp_h = (tid % CPT)*(CPB/(int) sizeof(half));

    // this thread's four (column, k-pair) slots, fixed for the whole loop. The B fragment
    // puts thread `lane` on column lane/4 of its 8-wide tile and on k-pair lane%4 (+4).
    const int q  = lane % 4;
    const int nl = lane / 4;

    // The four rows this thread needs for a column are r = 2q, 2q+1, 8+2q, 8+2q+1, and
    // pi() of those is 8q, 8q+1, 8q+2, 8q+3 -- consecutive. Their fields therefore start at
    // consecutive multiples of K in the same bit stream, so ONE 32-bit window holds all
    // four (3K + 16 <= 25 bits): one LDS.64 and one funnel shift per column feed four
    // weights, where the shared-memory version needed four of each.
    int pw0[NTT];
    int psh[NTT];
#pragma unroll
    for (int j2 = 0; j2 < NTT; ++j2) {
        const int ccl = j2*8 + nl;
        // start of the LOWEST of the four fields, pi = 8q + 3
        int sp = ((32 - K) - K*(escha_dep_pi(2*q) + 3 + 32*ccl + 4*(ccl >> 3))) % NB;
        if (sp < 0) {
            sp += NB;
        }
        const int g0 = sp >> 5;
        pw0[j2] = warp*ESCHA_MAX_W + (g0 ? (NWD - g0) : 0);
        psh[j2] = sp & 31;
    }

    typedef ggml_cuda_mma::tile<16, 8, float> tile_c;
    typedef ggml_cuda_mma::tile<16, 8, half2> tile_a;
    typedef ggml_cuda_mma::tile<8,  8, half2> tile_b;

    tile_c acc[MT][NTT];

    // one payload word per thread, held a tile ahead
    static_assert(NTJ*((16*K)/2) <= NT, "escha: payload needs more than one word per thread");
    const bool has_pay = tid < NTJ*n_wd;
    const int  pt = tid/n_wd, pw = tid % n_wd;
    uint32_t   ppre = 0;
    if (has_pay && lo < hi) {
        ppre = ((const uint32_t *)(code + (int64_t)(lo*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
    }

    if (lo < hi) {
        const int row = row0 + cp_m;
        const int src_row = row < n_rows ? row : 0;
        __pipeline_memcpy_async(s_u + cp_m*ESCHA_TILE + cp_h,
                                u + (int64_t) src_row*IC + lo*ESCHA_TILE + cp_h,
                                CPB, row < n_rows ? 0 : CPB);
        __pipeline_commit();
    }

    for (int ti = lo; ti < hi; ++ti) {
        const int    buf    = (ti - lo) & 1;
        uint2 * pay_cur     = s_pay + buf*(NTJ*ESCHA_MAX_W);
        half  * su_cur      = s_u   + buf*(BM*ESCHA_TILE);
        half  * su_nxt      = s_u   + (buf ^ 1)*(BM*ESCHA_TILE);

        if (has_pay) {
            // word pw is the high half of pair pw and the low half of pair pw+1
            pay_cur[pt*ESCHA_MAX_W + pw].y = ppre;
            pay_cur[pt*ESCHA_MAX_W + (pw + 1 == NWD ? 0 : pw + 1)].x = ppre;
        }
        if (has_pay && ti + 1 < hi) {
            ppre = ((const uint32_t *)(code + (int64_t)((ti + 1)*nct + oc0/ESCHA_TILE + pt)*(16*K)))[pw];
        }
        __pipeline_wait_prior(0);
        // the only barrier in the loop: everything written past it goes to the OTHER
        // buffer, so nothing downstream needs protecting from the next iteration
        __syncthreads();

        if (ti + 1 < hi) {
            const int row = row0 + cp_m;
            const int src_row = row < n_rows ? row : 0;
            __pipeline_memcpy_async(su_nxt + cp_m*ESCHA_TILE + cp_h,
                                    u + (int64_t) src_row*IC + (ti + 1)*ESCHA_TILE + cp_h,
                                    CPB, row < n_rows ? 0 : CPB);
            __pipeline_commit();
        }

        tile_b B[NTT];
#pragma unroll
        for (int j2 = 0; j2 < NTT; ++j2) {
            const uint2    p = pay_cur[pw0[j2]];
            const uint32_t w = __funnelshift_r(p.y, p.x, psh[j2]);
            // shift 3K, 2K, K, 0  <->  r = 2q, 2q+1, 8+2q, 8+2q+1
            B[j2].x[0] = __halves2half2(escha_codebook_h((w >> (3*K)) & 0xffffu),
                                        escha_codebook_h((w >> (2*K)) & 0xffffu));
            B[j2].x[1] = __halves2half2(escha_codebook_h((w >>    K ) & 0xffffu),
                                        escha_codebook_h( w           & 0xffffu));
        }

        {
            const half2 * su2 = (const half2 *) su_cur;   // [BM][8] half2
            tile_a A[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                ggml_cuda_mma::load_ldmatrix(A[i], su2 + (size_t)(i*16)*8, 8);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
#pragma unroll
                for (int j2 = 0; j2 < NTT; ++j2) {
                    ggml_cuda_mma::mma(acc[i][j2], A[i], B[j2]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j2 = 0; j2 < NTT; ++j2) {
#pragma unroll
            for (int l = 0; l < tile_c::ne; ++l) {
                const int m   = i*16 + tile_c::get_i(l);
                const int n   = warp*(8*NTT) + j2*8 + tile_c::get_j(l);
                const int row = row0 + m;
                if (row < n_rows) {
                    partial[((int64_t) sl*n_rows + row)*OC + oc0 + n] = acc[i][j2].x[l];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(code, u, partial, IC, OC, n_rows, n_slices);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
template <int K, int R, bool QUAD>
static __global__ void escha_matmul_dense(
        const int16_t * __restrict__ code,
        const half    * __restrict__ lut,
        const int16_t * __restrict__ dep,
        const float   * __restrict__ u,
        float         * __restrict__ partial,
        const int IC, const int OC, const int n_rows, const int n_slices) {
    extern __shared__ char s_raw[];

    // every decode wants the adjacent pair (pay[w0-1], pay[w0]), so hold the payload AS
    // overlapping pairs: one aligned LDS.64 then replaces the two LDS the funnel shift needed
    uint2 * s_pay = (uint2 *) s_raw;                                   // [ESCHA_GROUPS][ESCHA_MAX_W]
    float * s_u   = (float *)(s_pay + ESCHA_GROUPS*ESCHA_MAX_W);       // [R][16]

    GGML_UNUSED(lut);
    GGML_UNUSED(dep);

    const int NW = 8*K;          // 32-bit words in a tile payload
    const int NB = 32*NW;        // bits

    const int nit  = IC/ESCHA_TILE;
    const int nct  = OC/ESCHA_TILE;
    const int n_wd = (16*K)/2;

    const int tid   = threadIdx.x;
    const int start = blockIdx.x*R;
    const int nrow  = min(R, n_rows - start);

    // this block's share of the input tiles
    const int sl  = blockIdx.z;
    const int lo  = (int) (((int64_t) nit*sl)/n_slices);
    const int hi  = (int) (((int64_t) nit*(sl + 1))/n_slices);

    const int grp = tid / ESCHA_TILE;
    const int cc  = tid % ESCHA_TILE;
    const int tj  = blockIdx.y*ESCHA_GROUPS + grp;

    uint2 * pay = s_pay + grp*ESCHA_MAX_W;

    // start position for this thread's column, before the per-row term
    int s0 = ((32 - K) - K*(32*cc + 4*(cc >> 3))) % NB;
    if (s0 < 0) {
        s0 += NB;
    }

    // Quad decode. Rows 2q, 2q+1, 8+2q, 8+2q+1 have pi() = 8q, 8q+1, 8q+2, 8q+3 --
    // consecutive -- so their four 16-bit fields start at consecutive multiples of K in the
    // same bit stream, and one 32-bit window holds all four (3K + 16 <= 25 bits). Four of
    // this column's sixteen weights then cost one LDS.64 and one funnel shift instead of
    // four of each. The weights are stashed in r order and consumed in r order below, so
    // the fp32 accumulation sequence is exactly the one the scalar loop produced.
    int qw0[4], qsh[4];
    if constexpr (QUAD) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            // K*(8q + 3) <= 81 < NB, so one conditional add restores the range
            int sp = s0 - K*(8*q + 3);
            if (sp < 0) {
                sp += NB;
            }
            const int g0 = sp >> 5;
            qw0[q] = g0 ? (NW - g0) : 0;
            qsh[q] = sp & 31;
        }
    }

    float acc[R];
#pragma unroll
    for (int m = 0; m < R; ++m) {
        acc[m] = 0.0f;
    }
    // At R == 1 the block reads one row, so its entire slice of u fits in shared and is
    // staged once here -- the per-tile staging below then disappears, and with it the
    // block-wide barrier that ordered it. This is what their 10,240-byte allocation is.
    if constexpr (R == 1) {
        const float * u_row = u + (int64_t) start*IC + (int64_t) lo*ESCHA_TILE;
        const int n_stage = (hi - lo)*ESCHA_TILE;
        for (int j = tid; j < n_stage; j += ESCHA_NT) {
            s_u[j] = u_row[j];
        }
    }
    // payload words this thread owns, and the tile fetched one iteration ahead
    constexpr int NPW = (8*K + ESCHA_TILE - 1)/ESCHA_TILE;
    uint32_t pre[NPW];
    if (lo < hi) {
        const uint32_t * s0p = (const uint32_t *)(code + (int64_t)(lo*nct + tj)*(16*K));
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                pre[i] = s0p[wd];
            }
        }
    }
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        // as in the routed kernel: the block cooperates on one 16-wide slice of u per row,
        // and the 16 threads of a group then read the same entry, so it broadcasts
        if constexpr (R != 1) {
            for (int j = tid; j < R*ESCHA_TILE; j += ESCHA_NT) {
                const int m = j / ESCHA_TILE;
                const int r = j % ESCHA_TILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m)*IC + ti*ESCHA_TILE + r] : 0.0f;
            }
        }

        // publish the tile fetched last round, then issue the next fetch immediately: the
        // LDG then overlaps this tile's decode instead of stalling in front of it, which is
        // what the LDG -> dependent STS pair at the top of the loop was costing
#pragma unroll
        for (int i = 0; i < NPW; ++i) {
            const int wd = cc + i*ESCHA_TILE;
            if (wd < n_wd) {
                // word wd is the high half of pair wd and the low half of pair wd+1
                pay[wd].y = pre[i];
                pay[wd + 1 == NW ? 0 : wd + 1].x = pre[i];
            }
        }
        if (ti + 1 < hi) {
            const uint32_t * nxt = (const uint32_t *)(code + (int64_t)((ti + 1)*nct + tj)*(16*K));
#pragma unroll
            for (int i = 0; i < NPW; ++i) {
                const int wd = cc + i*ESCHA_TILE;
                if (wd < n_wd) {
                    pre[i] = nxt[wd];
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }

        const float * uu = s_u + (ti - lo)*ESCHA_TILE;

        if constexpr (QUAD) {
            half wh[ESCHA_TILE];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint2    p = pay[qw0[q]];
                const uint32_t w = __funnelshift_r(p.y, p.x, qsh[q]);
                // shift 3K, 2K, K, 0  <->  pi = 8q, 8q+1, 8q+2, 8q+3  <->  r below
                wh[2*q    ] = escha_codebook_h((w >> (3*K)) & 0xffffu);
                wh[2*q + 1] = escha_codebook_h((w >> (2*K)) & 0xffffu);
                wh[8 + 2*q] = escha_codebook_h((w >>    K ) & 0xffffu);
                wh[9 + 2*q] = escha_codebook_h( w           & 0xffffu);
            }
            // consumed in r order: escha_codebook is __half2float of escha_codebook_h, so
            // every product and every partial sum matches the scalar loop bit for bit
#pragma unroll
            for (int r = 0; r < ESCHA_TILE; ++r) {
                const float wv = __half2float(wh[r]);
                if constexpr (R == 1) {
                    acc[0] += uu[r]*wv;
                } else {
#pragma unroll
                    for (int m = 0; m < R; ++m) {
                        acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                    }
                }
            }
        } else {
            // the original scalar loop, kept so ESCHA_NO_ADAPT can select it: one payload
            // load and one funnel shift per weight. Unrolls fully at R <= 16 so pi(r) folds
            // to a compile-time constant; prefill keeps a partial unroll, where acc[R]
            // already claims the registers
#pragma unroll (R <= 16 ? 16 : 4)
            for (int r = 0; r < ESCHA_TILE; ++r) {
                // K*pi(r) <= 81 < NB, so one conditional add restores the range
                int sp = s0 - K*escha_dep_pi(r);
                if (sp < 0) {
                    sp += NB;
                }

                const int g0 = sp >> 5;
                const int w0 = g0 ? (NW - g0) : 0;
                const int w1 = w0 ? (w0 - 1)  : (NW - 1);

                const uint2 p = pay[w0];
                GGML_UNUSED(w1);
                const uint32_t idx = __funnelshift_r(p.y, p.x, sp & 31) & 0xffffu;

                const float wv = escha_codebook(idx);
                if constexpr (R == 1) {
                    acc[0] += uu[r]*wv;
                } else {
#pragma unroll
                    for (int m = 0; m < R; ++m) {
                        acc[m] += s_u[m*ESCHA_TILE + r]*wv;
                    }
                }
            }
        }
        if constexpr (R == 1) { __syncwarp(); } else { __syncthreads(); }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t) sl*n_rows + start + m)*OC + blockIdx.y*ESCHA_NT + tid] = acc[m];
    }
}

// sum the slices in a fixed order (so the result is reproducible), rotate the
// 128-column group, scale by rout
static __global__ void escha_finalize_dense(
        const half  * __restrict__ rout,
        const float * __restrict__ partial,
        float       * __restrict__ dst,
        const int OC, const int ne1, const int n_rows, const int n_slices,
        const int64_t nb_d1, const int64_t nb_d2) {
    __shared__ float s_acc[ESCHA_NT];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y*ESCHA_NT + tid;

    float sum = 0.0f;
    for (int s = 0; s < n_slices; ++s) {
        sum += partial[((int64_t) s*n_rows + row)*OC + c];
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_hadamard_128(s_acc, ESCHA_NT, tid, ESCHA_NT);

    float * dst_row = (float *)((char *) dst + (int64_t)(row % ne1)*nb_d1
                                             + (int64_t)(row / ne1)*nb_d2);
    dst_row[c] = s_acc[tid]*__half2float(rout[c]);
}

void ggml_cuda_op_escha_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * code = dst->src[0];
    const ggml_tensor * rin  = dst->src[1];
    const ggml_tensor * rout = dst->src[2];
    const ggml_tensor * lut  = dst->src[3];
    const ggml_tensor * dep  = dst->src[4];
    const ggml_tensor * x    = dst->src[5];

    GGML_ASSERT(code->type == GGML_TYPE_I16 && dep->type == GGML_TYPE_I16);
    GGML_ASSERT(rin->type == GGML_TYPE_F16 && rout->type == GGML_TYPE_F16 && lut->type == GGML_TYPE_F16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int K   = code->ne[0]/16;
    const int OC  = code->ne[1]*16;
    const int IC  = code->ne[2]*16;
    const int nit = IC/ESCHA_TILE;

    const int n_rows = x->ne[1]*x->ne[2];
    const int n_ocb  = OC/ESCHA_NT;

    cudaStream_t stream = ctx.stream();

    // ESCHA_NO_ADAPT=1 reverts every change on this branch at runtime -- the adaptive mma
    // row tile, the register-B prefill kernel and the quad decode below -- so a variant and
    // its control can be gated from ONE binary in adjacent runs. Whole-run drift on this
    // card is a few percent, wider than some of the deltas being judged.
    static const bool no_adapt = getenv("ESCHA_NO_ADAPT") != nullptr;
    const bool quad = !no_adapt;

    // The tensor-core path wants its activations already in fp16 so cp.async can move them
    // verbatim, so the rotation has to know its consumer before it runs.
    const bool gen = n_rows <= ESCHA_GEN_MAX_ROWS;
    const bool use_mma = !gen
                      && ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_TURING
                      && OC % ESCHA_MMA_BN == 0
                      && getenv("ESCHA_NO_MMA") == nullptr;

    ggml_cuda_pool_alloc<char> u_buf(ctx.pool(),
        (size_t) n_rows*IC*(use_mma ? sizeof(half) : sizeof(float)));

    // one warp per independent 128-block instead of one block per row -- see
    // escha_rotate_in_dense_warp. Falls back to the per-row kernel if IC is not a whole
    // number of 128-blocks (which the codec's T128 implies, but the check is cheap).
    // ESCHA_NO_ROTWARP=1 pins the rotation back to the one-block-per-row kernel, i.e.
    // exactly the pre-change behaviour, so the wide rotate and its control can be gated
    // from ONE binary in adjacent runs (whole-run drift here is a few percent).
    static const bool no_rotwarp = getenv("ESCHA_NO_ROTWARP") != nullptr;
    if (IC % 128 == 0 && !no_rotwarp) {
        const int     hb_per_row = IC/128;
        const int64_t n_hb       = (int64_t) n_rows*hb_per_row;
        const int     nwarps     = ESCHA_ROT_NT/32;
        const int64_t nblk       = (n_hb + nwarps - 1)/nwarps;
        const int     grid       = (int) MIN(nblk, (int64_t) 65535);
        if (use_mma) {
            escha_rotate_in_dense_warp<half><<<grid, ESCHA_ROT_NT, 0, stream>>>(
                (const half *) rin->data, (const float *) x->data, (half *) u_buf.get(),
                IC, (int) x->ne[1], hb_per_row, n_hb, x->nb[1], x->nb[2]);
        } else {
            escha_rotate_in_dense_warp<float><<<grid, ESCHA_ROT_NT, 0, stream>>>(
                (const half *) rin->data, (const float *) x->data, (float *) u_buf.get(),
                IC, (int) x->ne[1], hb_per_row, n_hb, x->nb[1], x->nb[2]);
        }
    } else if (use_mma) {
        escha_rotate_in_dense<half><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (half *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    } else {
        escha_rotate_in_dense<float><<<n_rows, 256, 0, stream>>>(
            (const half *) rin->data, (const float *) x->data, (float *) u_buf.get(),
            IC, (int) x->ne[1], x->nb[1], x->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());

    // slice the IC reduction only as far as it takes to fill the device: at batch 1 the
    // natural grid is just n_ocb blocks, but a long prompt already has plenty of rows
    const int  R   = gen ? escha_gen_rows(n_rows) : ESCHA_ROWS_DENSE;

    const int n_rb = (n_rows + R - 1)/R;
    // the tiled prefill kernel blocks over BM rows x BN columns instead
    const int n_tb = (n_rows + ESCHA_BM - 1)/ESCHA_BM;
    const int n_cb = OC/ESCHA_BN;
    // batch 1 has only n_ocb blocks before slicing (136 for the FFN), which leaves an 82-SM
    // device mostly idle, so generation slices the reduction much harder than prefill
    const int target = gen ? ESCHA_GEN_TARGET_MUL*ESCHA_TARGET : ESCHA_TARGET;
    // NOTE: the gen slice count is deliberately derived from n_rows, NOT from n_rb.
    // The slice boundaries decide how the fp32 partials are grouped before
    // escha_finalize_dense adds them, so they are part of the ARITHMETIC, not just the
    // launch shape. Deriving them from n_rb (which the row tile drops to 1) would regroup
    // the sum and change the result bit for bit. Keeping the original expression makes a
    // row tile of R produce exactly what R separate one-row blocks produced -- the decode
    // is shared, the arithmetic is untouched.
    int n_slices = target/MAX(1, gen ? n_rows*n_ocb : n_tb*n_cb);
    n_slices = MIN(MAX(n_slices, 1), nit);

    ggml_cuda_pool_alloc<float> p_buf(ctx.pool(), (size_t) n_slices*n_rows*OC);

    static const bool no_gentile = getenv("ESCHA_NO_GENTILE") != nullptr;

    if (gen && R >= ESCHA_GEN_TILE_MIN && OC % ESCHA_BN == 0 && !no_gentile) {
        // Rows-per-thread refactor of the generation path.
        //
        // escha_matmul_dense gives every thread ALL R rows, so acc[R] is per thread and the
        // block is pinned at ESCHA_NT/32 = 4 warps however much work there is. From n_rows
        // 8 upwards that is the binding constraint: n_slices has already collapsed to 1
        // (bit-exactness pins it to n_rows) and n_rb is 1, so the FFN launches n_ocb = 136
        // blocks = 544 warps on a 36-SM device -- 15 warps per SM out of 48, against 36 at
        // batch 1, where the reduction is sliced 15 ways.
        //
        // Splitting the COLUMN axis does NOT recover that. Columns are already one per
        // thread, so the launch holds OC*n_slices threads whatever the block shape is; a
        // narrower block just cuts the same 544 warps into more, smaller blocks. Block
        // count is not occupancy.
        //
        // What does add warps is giving each thread FEWER rows and more threads to the
        // block: a BM x BN block with a TM x TN register tile per thread has
        // NT = (BM/TM)*(BN/TN) threads, so BM=R, BN=128, TN=2 is 8 warps instead of 4 and
        // acc shrinks from R to TM*TN. The decoded weights go to SHARED, so a weight is
        // still decoded exactly once per block -- unlike splitting the row axis across
        // blocks, this buys warps without duplicating any decode.
        //
        // But it is NOT free, and the measurement is the point of this comment. A decoded
        // weight in escha_matmul_dense lives in a register and feeds all R accumulators of
        // the owning thread; here it makes a round trip through shared (STS then LDS) to
        // feed only TM*TN. Per input tile, per row, the scalar kernel issues 122 warp
        // instructions and this one issues 137 -- 12% MORE. Doubling warps per SM (15.1 ->
        // 30.2 on the 136-block FFN launch) bought back rather less than that costs at R=8
        // and rather more at R=16:
        //
        //     batch 16, R=16:  86.4 -> 94.1 tok/s   (+8.9%, +12% instructions)
        //     batch  8, R=8 :  77.7 -> 53.9 tok/s   (-31%,  +36% instructions)
        //
        // The batch-16 figure is an A/B of the SAME binary in adjacent gate runs, via
        // ESCHA_NO_GENTILE -- run-to-run drift on this card is +-2-5%, wide enough that a
        // 9% claim needs the control rather than a comparison against an older run.
        //
        // So the generation path is issue-bound, not occupancy-bound, over this whole range:
        // runtime tracks instruction count far more closely than it tracks warps. Occupancy
        // is worth roughly 20% here, and only enough to pay for the extra instructions once
        // a thread owns 8 outputs. Do not widen ESCHA_GEN_TILE_MIN without re-measuring.
        //
        // Bit-exact because it changes only WHICH thread owns an output, never the order a
        // given output is summed in. Both kernels run ti ascending, then r ascending, over
        // the same lo/hi slice, over the same fp32 u and the same escha_codebook weight.
        // n_slices stays derived from n_rows, untouched.
        constexpr int NTJ = ESCHA_BN/ESCHA_TILE;
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) ESCHA_TILE*(ESCHA_BN + 2)*sizeof(float)
                          + (size_t) 2*ESCHA_TILE*R*sizeof(float);
        GGML_ASSERT(smem <= 48*1024 && "escha: gen row tile exceeds the default shared budget");
        auto launch = [&](auto kernel, int nt) {
            kernel<<<dim3(n_rb, OC/ESCHA_BN, n_slices), nt, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        // TN stays 2: at TN=1 every shared read feeds exactly one MAC and shared bandwidth
        // becomes the ceiling. NT stays 256 so one payload word per thread still covers
        // K=3 (NTJ*24 = 192 <= NT), and so the decode mapping has one thread per
        // (r, column-in-tile) pair.
        //
        // Only R=16 is instantiated. <KK, 8, ESCHA_BN, 2, 2> also compiles and is bit-exact
        // but measured 31% slower than escha_matmul_dense<KK,8>; ESCHA_GEN_TILE_MIN gates it.
#define ESCHA_GENTILE_LAUNCH(KK)                                                            \
        switch (R) {                                                                        \
            case 16: launch((escha_matmul_dense_tiled<KK, 16, ESCHA_BN, 4, 2>), 256); break; \
            default: GGML_ABORT("escha: unsupported gen tile R=%d", R);                     \
        }
        switch (K) {
            case 2: ESCHA_GENTILE_LAUNCH(2); break;
            case 3: ESCHA_GENTILE_LAUNCH(3); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
#undef ESCHA_GENTILE_LAUNCH
    } else if (gen) {
        // widest slice any block gets, since lo/hi split nit unevenly by at most one tile
        const int tiles_max = (nit + n_slices - 1)/n_slices;
        // R == 1 stages its entire slice of u once; R > 1 stages one [R][16] tile per step
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t)(R == 1 ? tiles_max : R)*ESCHA_TILE*sizeof(float);
        GGML_ASSERT(smem <= 48*1024 && "escha: staged u exceeds the default shared budget");
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_rb, n_ocb, n_slices), ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
#define ESCHA_GEN_LAUNCH(KK, QQ)                                              \
        switch (R) {                                                          \
            case  1: launch((escha_matmul_dense<KK,  1, QQ>)); break;          \
            case  2: launch((escha_matmul_dense<KK,  2, QQ>)); break;          \
            case  4: launch((escha_matmul_dense<KK,  4, QQ>)); break;          \
            case  8: launch((escha_matmul_dense<KK,  8, QQ>)); break;          \
            case 16: launch((escha_matmul_dense<KK, 16, QQ>)); break;          \
            default: GGML_ABORT("escha: unsupported gen R=%d", R);            \
        }
#define ESCHA_GEN_LAUNCH_Q(KK)                                                \
        if (quad) { ESCHA_GEN_LAUNCH(KK, true) } else { ESCHA_GEN_LAUNCH(KK, false) }
        switch (K) {
            case 2: ESCHA_GEN_LAUNCH_Q(2); break;
            case 3: ESCHA_GEN_LAUNCH_Q(3); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
#undef ESCHA_GEN_LAUNCH_Q
#undef ESCHA_GEN_LAUNCH
    } else if (OC % ESCHA_BN != 0) {
        // the tiled kernel blocks the output axis in exact BN steps; a ragged OC would
        // silently leave the tail columns unwritten. Every projection in this checkpoint is
        // 128-aligned, but that is a property of the model, not of the format.
        const size_t smem = ESCHA_GROUPS*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) ESCHA_ROWS_DENSE*ESCHA_TILE*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3((n_rows + ESCHA_ROWS_DENSE - 1)/ESCHA_ROWS_DENSE, n_ocb, n_slices),
                     ESCHA_NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch((escha_matmul_dense<2, ESCHA_ROWS_DENSE, false>)); break;
            case 3: launch((escha_matmul_dense<3, ESCHA_ROWS_DENSE, false>)); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    } else if (use_mma) {
        // tensor-core prefill. Weights are exact; activations are rounded to fp16, which is
        // what escha's runtime does. ESCHA_NO_MMA=1 falls back to the fp32 FMA kernel.
        constexpr int NTJ = ESCHA_MMA_BN/ESCHA_TILE;
        // ESCHA_NO_ADAPT=1 pins the row tile back at 128 and takes the original shared-memory
        // kernel, i.e. exactly the pre-change behaviour. It exists so a variant and its control
        // can be gated from ONE binary in adjacent runs: whole-run drift on this card is a few
        // percent, which is wider than some of the deltas being judged.
        // The row tile is picked from the batch, not fixed at 128. A block decodes 16 x BN
        // weights per input tile whatever BM is, so as long as BM still covers the batch in
        // one block the decode is exactly the same work -- but the tensor cores, the A-side
        // ldmatrix traffic and the accumulator registers all scale with BM, and at batch 32
        // a BM of 128 spent 3/4 of them on padding rows.
        // n_slices above stays derived from ESCHA_BM, so the fp32 partial grouping -- which
        // is arithmetic, not launch shape -- is untouched.
        const int bm = no_adapt ? ESCHA_MMA_BM
                     : (n_rows <= 32 ? 32 : (n_rows <= 64 ? 64 : ESCHA_MMA_BM));
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) 2*bm*ESCHA_TILE*sizeof(half)
                          + (size_t) ESCHA_MMA_BN*ESCHA_TILE*sizeof(half);
        const int n_tb_mma = (n_rows + bm - 1)/bm;
        const int n_cb_mma = OC/ESCHA_MMA_BN;
        if (bm < ESCHA_MMA_BM) {
            // narrow row tiles go to the register-B kernel: it decodes into the mma
            // fragments directly, so there is no s_w round trip and no B-side ldmatrix
            const size_t smem_rb = 2*NTJ*ESCHA_MAX_W*sizeof(uint2)
                                 + (size_t) 2*bm*ESCHA_TILE*sizeof(half);
            auto launch_rb = [&](auto kernel) {
                kernel<<<dim3(n_tb_mma, n_cb_mma, n_slices), dim3(32, 256/32), smem_rb, stream>>>(
                    (const int16_t *) code->data, (const half *) u_buf.get(), p_buf.get(),
                    IC, OC, n_rows, n_slices);
            };
#define ESCHA_RB_LAUNCH(KK)                                                        \
            switch (bm) {                                                          \
                case 32: launch_rb((escha_matmul_dense_mma_rb<KK, 32, ESCHA_MMA_BN, 4>)); break; \
                case 64: launch_rb((escha_matmul_dense_mma_rb<KK, 64, ESCHA_MMA_BN, 3>)); break; \
                default: GGML_ABORT("escha: unsupported rb BM=%d", bm);            \
            }
            switch (K) {
                case 2: ESCHA_RB_LAUNCH(2); break;
                case 3: ESCHA_RB_LAUNCH(3); break;
                default: GGML_ABORT("escha: unsupported K=%d", K);
            }
#undef ESCHA_RB_LAUNCH
            CUDA_CHECK(cudaGetLastError());
            escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
                (const half *) rout->data, p_buf.get(), (float *) dst->data,
                OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb_mma, n_cb_mma, n_slices), dim3(32, 256/32), smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const half *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        // warp split per row tile, chosen to minimise ldmatrix re-reads:
        // shared traffic is WN*(A bytes) + WM*(B bytes), so wide tiles want more warp rows.
#define ESCHA_MMA_LAUNCH(KK)                                                                 \
        switch (bm) {                                                                        \
            case  32: launch((escha_matmul_dense_tiled_mma<KK,  32, ESCHA_MMA_BN, 1>)); break;\
            case  64: launch((escha_matmul_dense_tiled_mma<KK,  64, ESCHA_MMA_BN, 2>)); break;\
            case 128: launch((escha_matmul_dense_tiled_mma<KK, 128, ESCHA_MMA_BN, 4>)); break;\
            default: GGML_ABORT("escha: unsupported mma BM=%d", bm);                         \
        }
        switch (K) {
            case 2: ESCHA_MMA_LAUNCH(2); break;
            case 3: ESCHA_MMA_LAUNCH(3); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
#undef ESCHA_MMA_LAUNCH
    } else {
        constexpr int NT  = (ESCHA_BM/ESCHA_TM)*(ESCHA_BN/ESCHA_TN);
        constexpr int NTJ = ESCHA_BN/ESCHA_TILE;
        const size_t smem = NTJ*ESCHA_MAX_W*sizeof(uint2)
                          + (size_t) ESCHA_TILE*(ESCHA_BN + 2)*sizeof(float)
                          + (size_t) 2*ESCHA_TILE*ESCHA_BM*sizeof(float);
        auto launch = [&](auto kernel) {
            kernel<<<dim3(n_tb, n_cb, n_slices), NT, smem, stream>>>(
                (const int16_t *) code->data, (const half *) lut->data, (const int16_t *) dep->data,
                (const float *) u_buf.get(), p_buf.get(), IC, OC, n_rows, n_slices);
        };
        switch (K) {
            case 2: launch((escha_matmul_dense_tiled<2, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            case 3: launch((escha_matmul_dense_tiled<3, ESCHA_BM, ESCHA_BN, ESCHA_TM, ESCHA_TN>)); break;
            default: GGML_ABORT("escha: unsupported K=%d", K);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    escha_finalize_dense<<<dim3(n_rows, n_ocb), ESCHA_NT, 0, stream>>>(
        (const half *) rout->data, p_buf.get(), (float *) dst->data,
        OC, (int) x->ne[1], n_rows, n_slices, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
