"""Frame-offset scenario: J486 = consistent scenario's J486 poses, but the J486 job's map frame is translated by
delta (map px, perpendicular to u) relative to J487.  Every J486 shift (concordant AND swap rows) gets + P_i delta.
The molecule's 2-fold and the refinements' swap both sit at c_true; only the J486 frame differs.
Usage: make_frame.py SRC_SCENARIO_DIR OUT_POSE_PKL DELTA_PX  [noise_px]"""
import json, pickle, sys
import numpy as np

src, out, dpx = sys.argv[1], sys.argv[2], float(sys.argv[3])
noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
t = json.load(open(f"{src}/truth.json"))
D, s_t = t["D"], t["s_t"]
u = np.array([0.720, -0.692, 0.061]); u /= np.linalg.norm(u)
e1 = np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(u, e1)
r6, t6 = pickle.load(open(f"{src}/proj/pseudo_c2_check/pose_J486_in_J487_order.pkl", "rb"))
r6 = np.asarray(r6, np.float64); t6 = np.asarray(t6, np.float64)
delta = dpx * e2
shift_px = np.einsum("nij,j->ni", r6[:, :2, :], delta)      # P_i delta, using J486's own rotation (its frame)
t6n = t6 + s_t * shift_px / (D + 1)
if noise > 0:
    rng = np.random.default_rng(5)
    t6n = t6n + s_t * rng.normal(0, noise, t6n.shape) / (D + 1)
pickle.dump((r6.astype(np.float32), t6n.astype(np.float32)), open(out, "wb"))
print("wrote", out, "delta px", delta.tolist(), "noise", noise)
