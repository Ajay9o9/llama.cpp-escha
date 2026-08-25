// Checks ggml_escha_linear against numpy references built by
// tools/escha-dump-linear-cases.py. The reference decodes the code to a dense
// matrix and does a plain matmul, sharing no code with the op under test.
//
//   usage: test-escha-linear [path-to-escha-linear-cases.gguf]

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const char * DEFAULT_CASES = "escha-linear-cases.gguf";

int main(int argc, char ** argv) {
    const char * fname = argc > 1 ? argv[1] : DEFAULT_CASES;

    // load every tensor eagerly into host memory
    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &ctx_data,
    };
    gguf_context * meta = gguf_init_from_file(fname, params);
    if (!meta || !ctx_data) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
        return 1;
    }

    std::vector<std::string> prefixes;
    const int n_tensors = gguf_get_n_tensors(meta);
    for (int i = 0; i < n_tensors; ++i) {
        std::string name = gguf_get_tensor_name(meta, i);
        size_t dot = name.find(".escha.");
        if (dot != std::string::npos) {
            std::string pfx = name.substr(0, dot);
            if (std::find(prefixes.begin(), prefixes.end(), pfx) == prefixes.end()) {
                prefixes.push_back(pfx);
            }
        }
    }
    printf("%s: %zu cases\n", __func__, prefixes.size());
    fflush(stdout);

    int n_failed = 0;
    const int n_threads = std::min(4, (int) std::thread::hardware_concurrency());

    for (const auto & pfx : prefixes) {
        auto get = [&](const char * key) -> ggml_tensor * {
            const std::string full = pfx + "." + key;
            return ggml_get_tensor(ctx_data, full.c_str());
        };

        ggml_tensor * code  = get("escha.code");
        ggml_tensor * rin   = get("escha.rin");
        ggml_tensor * rout  = get("escha.rout");
        ggml_tensor * s_in  = get("escha.s_in");
        ggml_tensor * s_out = get("escha.s_out");
        ggml_tensor * bias  = get("escha.bias");
        ggml_tensor * x     = get("escha.x");
        ggml_tensor * yref  = get("escha.yref");

        GGML_ASSERT(code && rin && rout && s_in && s_out && bias && x && yref);

        const int64_t IC = ggml_nelements(rin);
        const int64_t OC = ggml_nelements(rout);

        // reshape the flat code into [n_code, OC/16, IC/16]
        const int64_t n_code = ggml_nelements(code)/((OC/16)*(IC/16));
        GGML_ASSERT(n_code == 32 || n_code == 48);


        // graph in a scratch context that also owns the result buffer;
        // weight tensors are shared with ctx_data
        struct ggml_init_params ip = {
            2*ggml_tensor_overhead()*1024 + ggml_graph_overhead() + (size_t)(OC*x->ne[1])*sizeof(float)*2,
            NULL, false };
        struct ggml_context * ctx0 = ggml_init(ip);

        const int n_th = n_threads > 0 ? n_threads : 4;
        printf("%s: computing %s (threads %d)\n", __func__, pfx.c_str(), n_th);
        fflush(stdout);

        ggml_tensor * out = ggml_escha_linear(ctx0, ggml_reshape_3d(ctx0, code, n_code, OC/16, IC/16), rin, rout, s_in, s_out, bias, x);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, out);
        printf("%s: graph built, computing\n", __func__);
        fflush(stdout);
        ggml_graph_compute_with_ctx(ctx0, gf, n_th);
        printf("%s: computed\n", __func__);
        fflush(stdout);

        float * out_data = (float *) malloc(ggml_nbytes(out));
        memcpy(out_data, out->data, ggml_nbytes(out));

        double max_abs = 0.0;
        double max_rel = 0.0;
        const float * ref = (const float *) yref->data;
        for (int64_t r = 0; r < x->ne[1]; ++r) {
            for (int64_t c = 0; c < OC; ++c) {
                const double g = out_data[r*OC + c];
                const double w = ref[r*OC + c];
                const double d = fabs(g - w);
                max_abs = d > max_abs ? d : max_abs;
                const double denom = fabs(w) > 1e-4 ? fabs(w) : 1e-4;
                const double rel = d/denom;
                max_rel = rel > max_rel ? rel : max_rel;
            }
        }

        // the reference rounds through fp16 while the op stays in fp32, so the
        // tolerance scales with the output magnitude rather than per-element
        double yscale = 0.0;
        for (int64_t i = 0; i < OC*x->ne[1]; ++i) {
            yscale = fabs(ref[i]) > yscale ? fabs(ref[i]) : yscale;
        }
        const double atol = 5e-3*yscale + 1e-2;

        printf("%s: IC=%" PRId64 " OC=%" PRId64 " M=%" PRId64
               "  max_abs=%.4g (atol %.4g, |y|max %.4g)\n",
               pfx.c_str(), IC, OC, x->ne[1], max_abs, atol, yscale);
        fflush(stdout);
        if (!(max_abs < atol)) {
            fprintf(stderr, "%s: FAIL %s\n", __func__, pfx.c_str());
            ++n_failed;
        }

        free(out_data);
        ggml_free(ctx0);
    }

    gguf_free(meta);
    ggml_free(ctx_data);
    printf("%s: %d failures\n", __func__, n_failed);
    fflush(stdout);
    return n_failed ? 1 : 0;
}
