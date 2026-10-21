"""Synthetic pseudo-dimer data set for testing linearity_check_cell.py with real cryodrgn backproject_voxel.

Molecule: two protomers related by a 2-fold about u through c_true (map frame, px from the box origin), plus an
"MPP6" blob on protomer A only. Images are analytic projections of Gaussian blobs (no interpolation error), with
CTF and noise. Particle types:
  conc  : J487 = J486 = truth
  swapA : J487 = truth,                  J486 = 2-fold swap (rotation r.C, shift moved for an axis at c_pose)
  swapB : J487 = 2-fold swap (c_pose),   J486 = truth
Usage: gen_lin.py OUTROOT SCENARIO   (SCENARIO in consistent | gap)
"""
import json, os, pickle, subprocess, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

out, scen = sys.argv[1], sys.argv[2]
D, APIX = 64, 2.609375
rng = np.random.default_rng(7)
u = np.array([0.720, -0.692, 0.061]); u /= np.linalg.norm(u)
C = 2 * np.outer(u, u) - np.eye(3)
e1 = np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(u, e1)
c_true = 1.2 * e1 + 0.6 * e2                           # molecule's 2-fold axis, px from the box origin
c_pose = c_true + (2.0 * e2 if scen == "gap" else 0)   # axis the "refinements" swap about
SIG = 1.6

a0 = c_true + 6.0 * e2
ptsA = a0 + rng.uniform(-1, 1, (12, 3)) * np.array([4.5, 4.5, 4.5])
ptsB = (ptsA - c_true) @ C.T + c_true
mpp6 = a0 + 7.5 * e2 + rng.uniform(-1, 1, (3, 3)) * 1.5
pts = np.vstack([ptsA, ptsB, mpp6])
wts = np.ones(len(pts))
assert np.abs(pts).max() < 22, np.abs(pts).max()

# truth map, array [z, y, x], origin at index D/2
g = np.arange(D) - D / 2
Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
V_true = sum(w * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) / (2 * SIG ** 2))
             for p, w in zip(pts, wts))

n_conc, n_A, n_B = 1800, 840, 360
N = n_conc + n_A + n_B
types = np.array(["conc"] * n_conc + ["swapA"] * n_A + ["swapB"] * n_B)
types = types[rng.permutation(N)]
r_true = Rot.random(N, random_state=11).as_matrix()     # map -> image, cryoDRGN convention
t_true = rng.uniform(-1.5, 1.5, (N, 2))                  # px

def ctf_img(D, apix):
    f = np.fft.fftfreq(D) / apix
    fy, fx = np.meshgrid(f, f, indexing="ij")
    s2 = fx ** 2 + fy ** 2
    volt, cs, w, df = 300e3, 2.7e7, 0.1, 15000.0
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
    gam = 2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
    return np.sqrt(1 - w ** 2) * np.sin(gam) - w * np.cos(gam)

CTF2 = ctf_img(D, APIX)
gi = np.arange(D) - D / 2
IY, IX = np.meshgrid(gi, gi, indexing="ij")

def project(r, t, sgn):
    q = pts @ r.T                                        # image coords of each blob (x_img = r . x_map)
    img = np.zeros((D, D))
    for (qx, qy, _), w in zip(q, wts):
        img += w * np.exp(-((IX - qx - sgn * t[0]) ** 2 + (IY - qy - sgn * t[1]) ** 2) / (2 * SIG ** 2))
    return img

def write_stack(path, sgn):
    from cryodrgn.mrcfile import write_mrc
    imgs = np.stack([project(r_true[i], t_true[i], sgn) for i in range(N)])
    imgs = np.real(np.fft.ifft2(np.fft.fft2(imgs) * CTF2))
    imgs += rng.normal(0, 0.6 * imgs.std(), imgs.shape)
    write_mrc(path, imgs.astype(np.float32), Apix=APIX)

os.makedirs(out, exist_ok=True)
proj = os.path.join(out, "proj"); stk = os.path.join(out, "stack")
for d in (proj, stk, os.path.join(proj, "pseudo_c2_check", "bp_J486"), os.path.join(proj, "backproject")):
    os.makedirs(d, exist_ok=True)
ctf = np.tile(np.array([D, APIX, 15000, 15000, 0, 300, 2.7, 0.1, 0], np.float32), (N, 1))
pickle.dump(ctf, open(os.path.join(proj, "ctf.pkl"), "wb"))
json.dump({"axis": {"body_modes": [{"axis": u.tolist()}]}}, open(os.path.join(proj, "pseudo_c2_check", "pc2_summary.json"), "w"))
write_stack(os.path.join(stk, "particles.mrcs"), +1)
open(os.path.join(stk, "particles.txt"), "w").write("particles.mrcs\n")

def bp(name, rot, tr, ind):
    pp = os.path.join(out, f"_cal_{name}.pkl"); ip = os.path.join(out, f"_cal_{name}_ind.pkl")
    pickle.dump((rot.astype(np.float32), tr.astype(np.float32)), open(pp, "wb"))
    pickle.dump(ind.astype(np.int64), open(ip, "wb"))
    od = os.path.join(out, "_cal", name)
    subprocess.run(["cryodrgn", "backproject_voxel", os.path.join(stk, "particles.txt"), "--poses", pp, "--ctf",
                    os.path.join(proj, "ctf.pkl"), "--ind", ip, "-o", od, "--uninvert-data", "--no-half-maps"],
                   check=True, capture_output=True)
    from cryodrgn.mrcfile import parse_mrc
    return np.asarray(parse_mrc(os.path.join(od, "backproject.mrc"))[0], float)

def cc(a, b):
    a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

# calibrate the translation sign: which pose-file shift reproduces the truth?
cal = np.nonzero(types == "conc")[0][:600]
s_t = max((+1, -1), key=lambda s: cc(bp(f"t{s}", r_true, s * t_true / (D + 1), cal), V_true))
print("translation sign", s_t, "corr(truth-pose map, V_true) =", round(cc(bp(f"t{s_t}", r_true, s_t * t_true / D, cal), V_true), 4))

# calibrate the sign of the shift change that makes a swapped assignment a good alignment (axis at c_true)
P = r_true[:, :2, :]
wt = (np.eye(3) - C) @ c_true
cal2 = np.nonzero(types != "conc")[0][:600]
rC = r_true @ C
s_w = max((+1, -1), key=lambda s: cc(bp(f"w{s}", rC, s_t * (t_true + s * (P @ wt)) / (D + 1), cal2), V_true))
print("swap shift sign", s_w)

wp = (np.eye(3) - C) @ c_pose
t_swap = t_true + s_w * (P @ wp)
noise_R = Rot.from_rotvec(rng.normal(size=(N, 3)) * np.radians(1.5)).as_matrix()
r7, t7 = r_true.copy(), t_true.copy()
r6, t6 = r_true.copy(), t_true.copy()
A, B = types == "swapA", types == "swapB"
r6[A] = rC[A] @ noise_R[A]; t6[A] = t_swap[A] + rng.normal(0, 0.2, (A.sum(), 2))
r7[B] = rC[B]; t7[B] = t_swap[B]
pickle.dump((r7.astype(np.float32), (s_t * t7 / (D + 1)).astype(np.float32)), open(os.path.join(proj, "pose.pkl"), "wb"))
pickle.dump((r6.astype(np.float32), (s_t * t6 / (D + 1)).astype(np.float32)),
            open(os.path.join(proj, "pseudo_c2_check", "pose_J486_in_J487_order.pkl"), "wb"))
ind_swap = np.nonzero(A | B)[0].astype(np.int64)
ind_conc = np.nonzero(types == "conc")[0].astype(np.int64)
pickle.dump(ind_swap, open(os.path.join(proj, "pseudo_c2_check", "bp_J486", "c2swap_J486.pkl"), "wb"))
pickle.dump(ind_conc, open(os.path.join(out, "conc_ind.pkl"), "wb"))
np.save(os.path.join(out, "V_true.npy"), V_true)
json.dump(dict(c_true=c_true.tolist(), c_pose=c_pose.tolist(), s_t=s_t, s_w=s_w, apix=APIX, D=D,
               n_swap=int(len(ind_swap))), open(os.path.join(out, "truth.json"), "w"))
print("wrote", out)
