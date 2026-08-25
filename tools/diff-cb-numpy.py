import re, json

NAME_OK = {'attn_norm','linear_attn_qkv_mixed','conv_output_raw','conv_output_silu',
 'q_conv_predelta','k_conv_predelta','v_conv_predelta','gate','beta_sigmoid',
 'attn_output','linear_attn_out','attn_residual','attn_post_norm',
 'ffn_gate_escha','ffn_up_escha','ffn_silu_par_escha','ffn_out','l_out',
 'Qcur_full','Qcur_normed','Kcur_normed','Qcur','Kcur','Vcur',
 'attn_pregate','attn_gated','result_norm'}

hdr = re.compile(r'common_debug_cb_eval:\s+(.+?)\s+=\s+\((\w+)\)\s+(\w+)\(.*=\s+\{([0-9,\s]+)\}')
sumre = re.compile(r'sum = (\-?[0-9.]+(?:e[-+][0-9]+)?)')

cpp = []
cur = None
import sys
CB=sys.argv[1] if len(sys.argv)>1 else '/tmp/cb.log'
for line in open(CB):
    m = hdr.search(line)
    if m:
        cur = [m.group(1), tuple(int(x) for x in m.group(4).split(',')), None]
        continue
    s = sumre.search(line)
    if s and cur is not None:
        cur[2] = float(s.group(1)); cpp.append(tuple(cur)); cur = None

def key_of(name):
    m = re.match(r'(.+?)-(\d+)$', name)
    if m and m.group(1) in NAME_OK:
        return f'{m.group(1)}#{m.group(2)}'
    return None

npd = {}
for name, v in json.load(open('/tmp/npy_sums.json')):
    npd[name] = npd.get(name, 0.0) + v

last = {}
order = []
for name, dims, sm in cpp:
    k = key_of(name)
    if k is None:
        continue
    if k not in last:
        order.append(k)
    last[k] = sm

bad = 0
print(f"{'node':30s} {'cpp_sum':>16s} {'npy_sum':>16s} {'rel_err':>10s}")
for k in order:
    sm = last[k]
    nv = npd.get(k)
    if nv is None:
        continue
    rel = abs(nv - sm) / max(abs(sm), 1e-9)
    flag = ''
    if rel > 5e-3:
        flag = '  <<< DIVERGE'; bad += 1
    print(f'{k:30s} {sm:16.4f} {nv:16.4f} {rel:10.2e}{flag}')
    if bad >= 30:
        break
