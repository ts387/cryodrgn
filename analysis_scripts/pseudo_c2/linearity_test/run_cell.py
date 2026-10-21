"""Execute linearity_check_cell.py with patched #@param values; dump key results to JSON.
Usage: run_cell.py CELL PARAMS_JSON OUT_JSON"""
import json, re, sys, traceback

cell = open(sys.argv[1]).read()
for k, v in json.loads(sys.argv[2]).items():
    cell, n = re.subn(rf"^{k} = .*$", lambda m: f"{k} = {v!r}", cell, flags=re.M)
    assert n == 1, k
ns = {"__name__": "__main__"}
out = {}
try:
    exec(compile(cell, "linearity_check_cell", "exec"), ns)
except RuntimeError as e:
    out["error"] = str(e)
    print("RuntimeError:", e)
keys = ["r_m7", "r_m6", "r_conv", "r_FS", "r_FSm", "r_6F", "r_67", "r_6S", "r_6C", "r_6V", "reg", "anti", "cS", "cV", "c_fr", "gap", "gap_fr", "frame_A", "se_c", "ok_V",
        "apix", "f_sw", "f_id", "res"]
for k in keys:
    if k not in ns:
        continue
    v = ns[k]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, list) and len(v) > 10:
        v = {"median": float(sorted(v)[len(v) // 2]), "n": len(v)}
    out[k] = v
json.dump(out, open(sys.argv[3], "w"), indent=1)
