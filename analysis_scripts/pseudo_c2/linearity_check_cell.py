#@title Linearity check: is the J486 map the 2-fold swap of the J487 map? { display-mode: "form" }
#@markdown Paste this into the **backproject notebook** and run it after cells 1.1–1.4 and 2.1–2.2 (cryoDRGN
#@markdown installed, stack on local disk). Your existing files are only read; new files go to `OUT_DIR`.
#@markdown
#@markdown 1. Pose check on the exact index file both swap maps used, including where the 2-fold axis must sit
#@markdown    for the J486 and J487 shifts to agree.
#@markdown 2. Three back-projections of the same particles (about 3 min each on a T4):
#@markdown    `J487_rerun` = pose.pkl as is; `J487xC_t487` = J487 rotations · C with J487 shifts, the exact swap of
#@markdown    M7 about the box origin; `J487xC_t486` = J487 rotations · C with J486 shifts, i.e. M6 without J486's
#@markdown    rotation scatter.
#@markdown 3. Correlations at `LOWPASS_A` between these, the maps Cowork analysed, and the swap operator
#@markdown    derived from the poses (no operator is fitted to the maps).
POSE_J487 = "/content/drive/MyDrive/mina53_mpp6/pose.pkl"  #@param {type:"string"}
POSE_J486 = "/content/drive/MyDrive/mina53_mpp6/pseudo_c2_check/pose_J486_in_J487_order.pkl"  #@param {type:"string"}
IND = "/content/drive/MyDrive/mina53_mpp6/pseudo_c2_check/bp_J486/c2swap_J486.pkl"  #@param {type:"string"}
SUMMARY = "/content/drive/MyDrive/mina53_mpp6/pseudo_c2_check/pc2_summary.json"  #@param {type:"string"}
CTF = "/content/drive/MyDrive/mina53_mpp6/ctf.pkl"  #@param {type:"string"}
PARTICLES = "/content/cryodrgn_work/stack/particles.128.txt"  #@param {type:"string"}
BP_DIR = "/content/drive/MyDrive/mina53_mpp6/backproject"  #@param {type:"string"}
V_NAME = "concordant_eq32638"  #@param {type:"string"}
M7_NAME = "c2swap_J487_eq32638"  #@param {type:"string"}
M6_NAME = "c2swap_J486_eq32638"  #@param {type:"string"}
OUT_DIR = "/content/drive/MyDrive/mina53_mpp6/backproject/linearity"  #@param {type:"string"}
LOCAL_DIR = "/content/cryodrgn_work/linearity"  #@param {type:"string"}
UNINVERT_DATA = True  #@param {type:"boolean"}
LOWPASS_A = 8.0  #@param {type:"number"}
BATCH = 8  #@param {type:"integer"}

import json, os, pickle, shutil, subprocess, sys, time
import numpy as np
from scipy import ndimage, optimize
from cryodrgn.mrcfile import parse_mrc


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _dump(obj, p):
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)


for p in (POSE_J487, POSE_J486, IND, SUMMARY, PARTICLES) + ((CTF,) if CTF else ()):
    if not os.path.exists(p):
        raise FileNotFoundError(p + ("   <- run cells 2.1-2.2 of the backproject notebook first"
                                     if p == PARTICLES else ""))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR, exist_ok=True)

# ------------------------------------------------------------------ 1. poses on the exact index set
u = np.asarray(json.load(open(SUMMARY))["axis"]["body_modes"][0]["axis"], float)
u /= np.linalg.norm(u)
C = 2 * np.outer(u, u) - np.eye(3)                     # 180 deg about u, map frame (x, y, z)
r7, t7 = (np.asarray(a, np.float64) for a in _load(POSE_J487))
r6, t6 = (np.asarray(a, np.float64) for a in _load(POSE_J486))
ind = np.asarray(_load(IND)).ravel().astype(np.int64)
if r7.shape != r6.shape or t7.shape != t6.shape or ind.max() >= len(r7):
    raise ValueError(f"pose files / index file do not fit together: {r7.shape} {r6.shape} {t7.shape} {t6.shape}, "
                     f"max index {ind.max()}")


def ang(Ra, Rb):                                       # rotation angle of Ra^T Rb, degrees
    c = (np.einsum("nji,nji->n", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


R7, R6 = r7[ind], r6[ind]
R7C = R7 @ C
a76, a6c = ang(R7, R6), ang(R7C, R6)
q = lambda a: f"median {np.median(a):6.1f}   5-95% {np.percentile(a, 5):5.1f}-{np.percentile(a, 95):5.1f}"
print(f"axis u = ({u[0]:+.4f}, {u[1]:+.4f}, {u[2]:+.4f});  {len(ind):,} particles in {os.path.basename(IND)}")
print(f"J486 vs J487     : {q(a76)} deg")
print(f"J486 vs J487 . C : {q(a6c)} deg;  {np.mean(a6c < 10):.1%} within 10 deg of the swap, "
      f"{np.mean(a76 < 10):.1%} within 10 deg of J487")

# Shifts. Pose translations are fractions of the box; report them in back-projection pixels and A.
v0, h0 = parse_mrc(os.path.join(BP_DIR, M7_NAME, "backproject.mrc"))
D, apix = v0.shape[0], float(h0.apix)
d = (t6[ind] - t7[ind]) * D                            # px at the back-projection box
dn = np.linalg.norm(d, axis=1)
print(f"|t486 - t487|    : median {np.median(dn):.2f} px ({np.median(dn) * apix:.1f} A), "
      f"95th {np.percentile(dn, 95):.2f} px  (box {D}, {apix:.4f} A/px)")

# If the molecule's 2-fold passes through c rather than the box origin, swapping a particle also moves it:
# d_i = P_i w with w = (I - C) c = 2 c_perp, P_i = first two rows of rot (cryoDRGN: x_image = rot . x_map).
# Fit w robustly (trimmed least squares). The control uses the transposed rotation, which has no physical meaning.
e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
e1 /= np.linalg.norm(e1)
Bp = np.stack([e1, np.cross(u, e1)], 1)                # basis of the plane perpendicular to u


def fit_shift(P):
    A = P @ Bp                                         # (n, 2, 2)
    sel = np.ones(len(d), bool)
    for _ in range(8):
        x = np.linalg.lstsq(A[sel].reshape(-1, 2), d[sel].reshape(-1), rcond=None)[0]
        res = np.linalg.norm(d - A @ x, axis=1)
        sel = res <= np.quantile(res, 0.8)
    return Bp @ x, res


w_px, res = fit_shift(R7[:, :2, :])
_, res_ctl = fit_shift(R7.transpose(0, 2, 1)[:, :2, :])
c_px = w_px / 2                                        # axis offset from the box origin, perpendicular to u
print(f"shift change explained by a 2-fold axis {np.linalg.norm(c_px) * apix:.1f} A from the box origin: "
      f"median residual {np.median(res):.2f} px (from {np.median(dn):.2f}; control {np.median(res_ctl):.2f})")

# ------------------------------------------------------------------ 2. pose files and back-projections
rC = np.einsum("nij,jk->nik", r7, C).astype(np.float32)
jobs = {"J487_rerun": POSE_J487,
        "J487xC_t487": os.path.join(OUT_DIR, "pose_J487xC_t487.pkl"),
        "J487xC_t486": os.path.join(OUT_DIR, "pose_J487xC_t486.pkl")}
for name, tr in (("J487xC_t487", t7), ("J487xC_t486", t6)):
    if not os.path.exists(jobs[name]):
        _dump((rC, tr.astype(np.float32)), jobs[name])
exe = shutil.which("cryodrgn") or os.path.join(os.path.dirname(sys.executable), "cryodrgn")
maps_dir = {}
for name, pose in jobs.items():
    drv, loc = os.path.join(OUT_DIR, name), os.path.join(LOCAL_DIR, name)
    maps_dir[name] = drv
    if os.path.exists(os.path.join(drv, "backproject.mrc")):
        print(f"{name}: already made, reusing {drv}")
        continue
    cmd = [exe, "backproject_voxel", PARTICLES, "--poses", pose, "--ind", IND, "-o", loc,
           "-b", str(int(BATCH)), "--reg-weight", "1.0", "--lazy"]
    if CTF:
        cmd += ["--ctf", CTF]
    if UNINVERT_DATA:
        cmd += ["--uninvert-data"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.exists(os.path.join(loc, "backproject.mrc")):
        print(" ".join(cmd), "\n", r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"{name}: backproject_voxel failed")
    shutil.copytree(loc, drv, dirs_exist_ok=True)
    print(f"{name}: back-projected in {(time.time() - t0) / 60:.1f} min -> {drv}")

# ------------------------------------------------------------------ 3. maps
def load(dr):
    v, h = parse_mrc(os.path.join(dr, "backproject.mrc"))
    return np.asarray(v, np.float64)


raw = {"V": load(os.path.join(BP_DIR, V_NAME)), "M7": load(os.path.join(BP_DIR, M7_NAME)),
       "M6": load(os.path.join(BP_DIR, M6_NAME)), "M7r": load(maps_dir["J487_rerun"]),
       "M7C": load(maps_dir["J487xC_t487"]), "F": load(maps_dir["J487xC_t486"])}
print("\nmap   min / max  (compare with the copies Cowork read)")
for k in ("V", "M7", "M6"):
    print(f"{k:4s} {raw[k].min():+.4f} / {raw[k].max():+.4f}")

k = np.fft.fftfreq(D, d=apix)
kk = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
filt = np.exp(-0.5 * kk * (LOWPASS_A / 1.6) ** 2)      # Gaussian, about 0.5 transmission at 1 / LOWPASS_A
mp = {n: np.real(np.fft.ifftn(np.fft.fftn(v) * filt)) for n, v in raw.items()}
env = ndimage.binary_dilation(mp["V"] > np.percentile(mp["V"], 90), iterations=3)


def corr(a, b, m=env):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


# The swap as an array operation. cryoDRGN maps are indexed [z, y, x] with the origin at index D/2.
Pm = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
Carr = Pm @ C @ Pm.T
vox = np.argwhere(env).astype(float)


def swap(vol, c=np.zeros(3)):                          # 2-fold about u through c (map-frame px from the origin)
    q0 = D / 2 + Pm @ np.asarray(c, float)
    return ndimage.affine_transform(vol, Carr, offset=q0 - Carr @ q0, order=1, mode="constant")


def corr_sw(a, b, c=np.zeros(3)):                      # corr(a, swap(b, c)) in the envelope, without a full resample
    q0 = D / 2 + Pm @ np.asarray(c, float)
    bs = ndimage.map_coordinates(b, ((vox - q0) @ Carr.T + q0).T, order=1, mode="constant")
    av = a[env] - a[env].mean()
    bs = bs - bs.mean()
    return float(av @ bs / np.sqrt((av @ av) * (bs @ bs)))


conv = corr_sw(mp["M7C"], mp["M7r"])
sgn = max((+1, -1), key=lambda s: corr_sw(mp["F"], mp["M7r"], s * c_px))
cS = sgn * c_px                                        # axis position implied by the J486 - J487 shifts
mp["S7"] = swap(mp["M7"], cS)                          # M7 swapped about that axis

names = ["V", "M7", "M7r", "M6", "F", "M7C", "S7"]
print(f"\ncorrelation at {LOWPASS_A:g} A inside the envelope ({env.sum():,} voxels):")
print("       " + "".join(f"{n:>8s}" for n in names))
for a in names:
    print(f"{a:>5s}  " + "".join(f"{corr(mp[a], mp[b]):8.4f}" for b in names))
print("M7r = pose.pkl re-run; F = J487.C rotations + J486 shifts; M7C = J487.C rotations + J487 shifts;\n"
      "S7 = M7 swapped (array operation) about the axis the J486 - J487 shifts imply.")

# V's own best 2-fold axis position (what register_fraction.py refines), for comparison with the pose-derived one.
best = min((optimize.minimize(lambda x: -corr_sw(mp["V"], mp["V"], Bp @ x), x0, method="Nelder-Mead",
                              options=dict(xatol=0.02, fatol=1e-6)) for x0 in (np.zeros(2), Bp.T @ cS)),
           key=lambda r: r.fun)
cV = Bp @ best.x
gap = np.linalg.norm(cV - cS) * apix

r_conv, r_m7 = conv, corr(mp["M7"], mp["M7r"])
r_6F, r_67, r_6S, r_6C = (corr(mp["M6"], mp[n]) for n in ("F", "M7", "S7", "M7C"))
r_FS = corr_sw(mp["F"], mp["M7r"], cS)
reg = corr(mp["M6"] - mp["M7"], mp["S7"] - mp["M7"])
print("\nCHECKS")
print(f"  a. array convention  corr(M7C, swap(M7r))   = {r_conv:.4f}   (~1: the swap operator below is exact)")
print(f"  b. M7 reproducible   corr(M7, M7r)          = {r_m7:.4f}   (~1: run 1 used pose.pkl on this set)")
print(f"  c. M6 as logged      corr(M6, F)            = {r_6F:.4f}   vs corr(M6, M7) = {r_67:.4f}")
print(f"  d. shifts = offset   corr(F, swap(M7r, c))  = {r_FS:.4f}   (axis {np.linalg.norm(cS) * apix:.1f} A "
      f"from the origin)")
print(f"  e. M6 vs swapped M7  corr(M6, S7) = {r_6S:.4f}   corr(M6, M7C) = {r_6C:.4f}   corr(M6, M7) = {r_67:.4f}")
print(f"  f. register          corr(M6 - M7, S7 - M7) = {reg:+.3f}   (+1: M6 is M7 swapped; ~0: it is not)")
print(f"  g. operator match    V's own 2-fold axis: {np.linalg.norm(cV) * apix:.1f} A from the origin, {gap:.1f} A "
      f"from the shift-derived axis;  corr(V, swap V) {-best.fun:.4f} there vs {corr_sw(mp['V'], mp['V'], cS):.4f}")

print("\nVERDICT")
if r_conv < 0.98:
    print("  The array convention check failed, so the operator comparisons are not interpretable. Send me the output.")
elif r_m7 < 0.995:
    print("  M7 on Drive is NOT pose.pkl applied to this index set: run 1 used other inputs. Use J487_rerun as M7.")
elif r_6F <= r_67:
    print("  M6 on Drive is closer to M7 than to the same particles with J487.C rotations and J486 shifts, so it is "
          "not what the run-2 log says. Send me the output.")
elif reg > 0.5:
    print(f"  Both maps are what their labels say, and M6 is M7 swapped about an axis {np.linalg.norm(cS) * apix:.1f} A "
          f"from the box origin (set by the J486 - J487 shift changes).")
    print(f"  V's own 2-fold axis is {gap:.1f} A from that one. " + (
          "The difference is large enough that a check of M6 against M7 with V's operator fails, which is what "
          "check (e) reported." if gap > 1.0 else "So check (e) should have passed; send me the output."))
else:
    print("  Both maps are what their labels say, but M6 is not M7 swapped about the shift-derived axis. Compare F "
          "with M7C (rows above) and send me the output.")
