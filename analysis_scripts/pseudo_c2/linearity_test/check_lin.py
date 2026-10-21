"""Assertions for the synthetic scenarios."""
import json, os, sys
import numpy as np

root = sys.argv[1]
fails = []


def chk(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


chk(open(os.path.join(root, "consistent_ok.log")).read().count("already made from these inputs") == 4,
    "consistent/ok: re-run reuses all four maps")
for sc, tag in (("consistent", "ok"), ("consistent", "mislabel7"), ("consistent", "mislabel6"),
                ("consistent", "wrongind"), ("consistent", "frame"), ("consistent", "blur"), ("gap", "ok")):
    r = json.load(open(os.path.join(root, f"{sc}_{tag}.json")))
    t = json.load(open(os.path.join(root, sc, "truth.json")))
    log = open(os.path.join(root, f"{sc}_{tag}.log")).read()
    if tag == "wrongind":
        chk("STOPPED before back-projecting" in log and "not (only) the c2_swap class" in log and "r_m7" not in r,
            f"{sc}/{tag}: pose gate stops before back-projecting and names the index file (f_sw {r.get('f_sw')})")
        continue
    chk("error" not in r and "VERDICT" in log, f"{sc}/{tag}: ran to the end")
    if "VERDICT" not in log:
        continue
    verdict = log.split("VERDICT", 1)[1]
    ap = r["apix"]
    ct, cp = np.array(t["c_true"]), np.array(t["c_pose"])
    cS, cV, cF = np.array(r["cS"]), np.array(r["cV"]), np.array(r["c_fr"])
    print(f"--- {sc}/{tag}: m7 {r['r_m7']:.4f} m6 {r['r_m6']:.4f} conv {r['r_conv']:.4f} FS {r['r_FS']:.4f}/{r['r_FSm']:.4f} "
          f"6S {r['r_6S']:.4f} 6V {r['r_6V']:.4f} 67 {r['r_67']:.4f} reg {r['reg']:+.3f} anti {r['anti']:+.3f} | "
          f"cS err {np.linalg.norm(cS - cp) * ap:.2f} c_fr err {np.linalg.norm(cF - cp) * ap:.2f} A (se {r['se_c']:.2f}), "
          f"cV err {np.linalg.norm(cV - ct) * ap:.2f} A, gap {r['gap']:.2f}/{r['gap_fr']:.2f} A, frame {r['frame_A']:.2f} A")
    chk(r["r_conv"] > 0.98, f"{sc}/{tag}: array convention ({r['r_conv']:.4f})")
    chk(r["r_FS"] > r["r_FSm"], f"{sc}/{tag}: convention sign confirmed by the maps")
    chk(np.linalg.norm(cF - cp) * ap < 0.8, f"{sc}/{tag}: refinements' swap axis within 0.8 A of the truth")
    chk(np.linalg.norm(cV - ct) * ap < 0.8, f"{sc}/{tag}: V's own axis within 0.8 A of the truth")
    chk("M6 is M7 swapped" in verdict, f"{sc}/{tag}: swap detected (reg {r['reg']:+.3f}, anti {r['anti']:+.3f})")
    chk("POOR FIT" not in log, f"{sc}/{tag}: no spurious poor-fit warning")
    m7ok, m6ok = tag != "mislabel7", tag != "mislabel6"
    chk((r["r_m7"] >= 0.999) == m7ok and (("M7 on Drive is NOT" in verdict) != m7ok), f"{sc}/{tag}: M7 provenance verdict")
    chk((r["r_m6"] >= 0.999) == m6ok and (("M6 on Drive is NOT" in verdict) != m6ok), f"{sc}/{tag}: M6 provenance verdict")
    if tag == "frame":
        chk(abs(r["frame_A"] - 2.0 * ap) < 0.5 and r["gap_fr"] < 1.0 and "Most of that gap is J486's frame offset" in verdict,
            f"{sc}/{tag}: frame offset measured ({r['frame_A']:.2f} A) and named as the cause")
    elif sc == "gap":
        chk(abs(r["gap"] - np.linalg.norm(cp - ct) * ap) < 1.0 and "therefore fails" in verdict
            and "shift-derived axis, not V's" in verdict, f"{sc}/{tag}: operator gap found and reported")
    else:
        want = "should have passed on these maps" if (m7ok and m6ok) else "should pass on the re-runs"
        chk(r["gap"] < 1.0 and (want in verdict or "blurrier" in verdict), f"{sc}/{tag}: no operator gap, verdict says so")
print("SUMMARY", "PASS" if not fails else f"FAIL ({len(fails)})")
