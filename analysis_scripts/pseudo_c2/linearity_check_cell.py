#@title Linearity check: is the J486 map the 2-fold swap of the J487 map? { display-mode: "form" }
#@markdown Paste this into the **backproject notebook** and run it after cells 1.1–1.4 and 2.1–2.2 (cryoDRGN
#@markdown installed, stack on local disk). Your existing files are only read. New maps go to `OUT_DIR`.
#@markdown
#@markdown 1. Pose check on the exact index file both swap maps used. It stops here if that set is not the
#@markdown    swap class. It also finds where the 2-fold axis must sit for the J486 and J487 shifts to agree,
#@markdown    and whether J486's map frame is offset from J487's (from the concordant particles).
#@markdown 2. Four back-projections of the same particles (about 3 min each on a T4):
#@markdown    `J487_rerun` and `J486_rerun` repeat the two swap maps from the pose files as they are now;
#@markdown    `J487xC_t487` = J487 rotations · C with J487 shifts, the exact swap of M7 about the box origin;
#@markdown    `J487xC_t486` = J487 rotations · C with J486 shifts, i.e. M6 without J486's rotation scatter.
#@markdown 3. Correlations, after the same `LOWPASS_A` filter `register_fraction.py` uses, between these maps,
#@markdown    the maps Cowork analysed, and M7 swapped about the axis the shifts imply.
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
#@markdown Must match the runs that made the maps (they used `--uninvert-data`).
UNINVERT_DATA = True  #@param {type:"boolean"}
LOWPASS_A = 8.0  #@param {type:"number"}
#@markdown Mass for the molecular mask (dimer plus MPP6), as in `register_fraction.py`.
MASK_KDA = 125.0  #@param {type:"number"}
BATCH = 8  #@param {type:"integer"}

import hashlib, json, os, pickle, shutil, subprocess, sys, time
import numpy as np
from scipy import ndimage, optimize
from cryodrgn.mrcfile import parse_mrc

CELL_VERSION = "3"


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _dump(obj, p):                                     # write-then-rename, so an interrupted write leaves nothing
    with open(p + ".tmp", "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(p + ".tmp", p)


maps_in = {n: os.path.join(BP_DIR, d, "backproject.mrc") for n, d in (("V", V_NAME), ("M7", M7_NAME), ("M6", M6_NAME))}
for p in (POSE_J487, POSE_J486, IND, SUMMARY, PARTICLES, *maps_in.values()) + ((CTF,) if CTF else ()):
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
a76, a6c = ang(R7, R6), ang(R7 @ C, R6)
sw = a6c < 60                                          # J486 ~ J487.C (the split's swap class reaches ~50 deg); a
                                                       # random rotation lands here 6% of the time
f_sw, f_id = sw.mean(), np.mean(a76 < 20)
f_no = 1 - f_sw - f_id
q = lambda a: f"median {np.median(a):6.1f}   5-95% {np.percentile(a, 5):5.1f}-{np.percentile(a, 95):5.1f}"
print(f"axis u = ({u[0]:+.4f}, {u[1]:+.4f}, {u[2]:+.4f});  {len(ind):,} particles in {os.path.basename(IND)}")
print(f"J486 vs J487     : {q(a76)} deg;  {f_id:.1%} within 20 deg")
print(f"J486 vs J487 . C : {q(a6c)} deg;  {f_sw:.1%} within 60 deg (swap-consistent);  {f_no:.1%} neither")
if f_sw < 0.8:
    why = (f"{f_id:.0%} have J486 ~ J487, so this index file is not (only) the c2_swap class. Both swap maps were "
           f"made from it, so neither is a map of swapped particles alone" if f_id >= f_no else
           f"{f_no:.0%} have J486 unrelated to J487, so the J486 pose file does not match these rows (wrong row order, "
           f"or not the J486 job). M7 may be fine; M6 was made with a mismatched pose file")
    raise RuntimeError(f"STOPPED before back-projecting: only {f_sw:.0%} of {os.path.basename(IND)} are "
                       f"swap-consistent; {why}. Send me this output.")

v0, h0 = parse_mrc(maps_in["M7"])
D, apix = v0.shape[0], float(h0.apix)
# Shifts: cryoDRGN multiplies the stored fraction by the symmetrised box D + 1 and shifts by that many pixels.
d = (t6[ind] - t7[ind]) * (D + 1)
dn = np.linalg.norm(d, axis=1)
print(f"|t486 - t487|    : median {np.median(dn[sw]):.2f} px ({np.median(dn[sw]) * apix:.1f} A), "
      f"95th {np.percentile(dn[sw], 95):.2f} px on the swap-consistent rows  (box {D}, {apix:.4f} A/px)")


def fit_lin(A, dd):                                    # trimmed least squares for dd_i = A_i x; A: (n, 2, k)
    sel = np.ones(len(dd), bool)
    for _ in range(8):
        x = np.linalg.lstsq(A[sel].reshape(-1, A.shape[2]), dd[sel].reshape(-1), rcond=None)[0]
        res = np.linalg.norm(dd - A @ x, axis=1)
        sel = res <= np.quantile(res, 0.8)
    return x, res


rng = np.random.default_rng(0)


def boot_se(A, dd, scale, n=30):                       # bootstrap standard error of scale * x, in A
    m = len(dd)
    bs = np.array([fit_lin(A[j], dd[j])[0] for j in (rng.integers(0, m, m) for _ in range(n))]) @ scale.T
    return float(np.sqrt(np.trace(np.cov(bs.T))) * apix)


# J486 vs J487 map-frame offset y, from concordant particles outside the index set: d_i = P_i y.
conc = ang(r7, r6) < 10
conc[ind] = False
ci = np.flatnonzero(conc)
ci = rng.choice(ci, min(len(ci), 100_000), replace=False) if len(ci) else ci
if len(ci) >= 100:
    y_px = fit_lin(r7[ci, :2, :], (t6[ci] - t7[ci]) * (D + 1))[0]
else:
    y_px = np.zeros(3)
frame_A = float(np.linalg.norm(y_px) * apix)
print(f"J486 - J487 map-frame offset from {len(ci):,} concordant particles: {frame_A:.2f} A")

# If the molecule's 2-fold passes through c rather than the box origin, swapping a particle also moves it:
# d_i = P_i w with w = (I - C) c = 2 c_perp and P_i = the first two rows of rot (cryoDRGN: x_image = rot . x_map).
# In cryoDRGN's sign convention (checked by back-projecting synthetic data with backproject_voxel) c = -w / 2; the
# maps check the sign again in section 3. A J486 frame offset y adds P_i C y on swapped rows.
e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
e1 /= np.linalg.norm(e1)
Bp = np.stack([e1, np.cross(u, e1)], 1)                # basis of the plane perpendicular to u
P7 = R7[sw, :2, :] @ Bp                                # (n, 2, 2)
x_s, res = fit_lin(P7, d[sw])
cS = -(Bp @ x_s) / 2                                   # axis relating M6 to M7, px from the box origin
se_c = boot_se(P7, d[sw], -Bp / 2)
x_f = fit_lin(P7, d[sw] - R7[sw, :2, :] @ (C @ y_px))[0]
c_fr = -(Bp @ x_f) / 2                                 # the refinements' swap axis, J486 frame offset removed
print(f"shift change explained by a 2-fold axis {np.linalg.norm(cS) * apix:.2f} +/- {se_c:.2f} A from the box "
      f"origin ({sw.sum():,} rows; per-particle scatter after the fit, median {np.median(res):.2f} px)")
if frame_A >= 0.5:
    print(f"   with J486's frame offset removed, the refinements' swap axis is {np.linalg.norm(c_fr) * apix:.2f} A "
          f"from the origin")

# ------------------------------------------------------------------ 2. back-projections (resumable)
rC = np.einsum("nij,jk->nik", r7, C).astype(np.float32)
ind_sha = hashlib.sha1(ind.tobytes()).hexdigest()


def _sig(p):
    return [os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))]


def _stack_sig(p):                                     # the list file and, for a .txt stack, the files it lists
    out = [_sig(p)]
    if p.endswith(".txt"):
        base = os.path.dirname(os.path.abspath(p))
        with open(p) as f:
            for ln in (x.strip() for x in f):
                if ln:
                    m = ln if os.path.isabs(ln) else os.path.join(base, ln)
                    out.append(_sig(m) if os.path.exists(m) else [m, None, None])
    return out


def _file_sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


jobs = {"J487_rerun": (POSE_J487, _file_sha(POSE_J487)),
        "J486_rerun": (POSE_J486, _file_sha(POSE_J486))}
for name, tr in (("J487xC_t487", t7), ("J487xC_t486", t6)):
    pth = os.path.join(LOCAL_DIR, f"pose_{name}.pkl")
    obj = (rC, tr.astype(np.float32))
    _dump(obj, pth)
    jobs[name] = (pth, hashlib.sha1(obj[0].tobytes() + obj[1].tobytes()).hexdigest())
exe = shutil.which("cryodrgn") or os.path.join(os.path.dirname(sys.executable), "cryodrgn")
stack_sig = _stack_sig(PARTICLES)
ctf_sig = _sig(CTF) if CTF else None
maps_dir = {}
for i, (name, (pose, pose_sha)) in enumerate(jobs.items(), 1):
    drv, loc = os.path.join(OUT_DIR, name), os.path.join(LOCAL_DIR, name)
    maps_dir[name] = drv
    key = json.loads(json.dumps(dict(cell=CELL_VERSION, poses=pose_sha, ind=ind_sha, ctf=ctf_sig, stack=stack_sig,
                                     uninvert=bool(UNINVERT_DATA), batch=int(BATCH))))
    man = os.path.join(drv, "inputs.json")             # written last: a folder without it is incomplete
    try:
        same = json.load(open(man)) == key and os.path.exists(os.path.join(drv, "backproject.mrc"))
    except (OSError, ValueError):
        same = False
    if same:
        print(f"[{i}/4] {name}: already made from these inputs, reusing {drv}")
        continue
    cmd = [exe, "backproject_voxel", PARTICLES, "--poses", pose, "--ind", IND, "-o", loc,
           "-b", str(int(BATCH)), "--reg-weight", "1.0", "--lazy"]
    if CTF:
        cmd += ["--ctf", CTF]
    if UNINVERT_DATA:
        cmd += ["--uninvert-data"]
    print(f"[{i}/4] {name}: back-projecting {len(ind):,} particles ...", flush=True)
    shutil.rmtree(loc, ignore_errors=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.exists(os.path.join(loc, "backproject.mrc")):
        print(" ".join(cmd), "\n", r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"{name}: backproject_voxel failed (exit {r.returncode})")
    shutil.rmtree(drv, ignore_errors=True)
    shutil.copytree(loc, drv)
    with open(man + ".tmp", "w") as f:
        json.dump(key, f)
    os.replace(man + ".tmp", man)
    print(f"        done in {(time.time() - t0) / 60:.1f} min -> {drv}", flush=True)

# ------------------------------------------------------------------ 3. maps
def load(path):
    return np.asarray(parse_mrc(path)[0], np.float64)


raw = {n: load(p) for n, p in maps_in.items()}
for n, j in (("M7r", "J487_rerun"), ("M6r", "J486_rerun"), ("M7C", "J487xC_t487"), ("F", "J487xC_t486")):
    raw[n] = load(os.path.join(maps_dir[j], "backproject.mrc"))
print("\nmap   min / max  (the copies Cowork read should match)")
for k in ("V", "M7", "M6"):
    print(f"{k:4s} {raw[k].min():+.4f} / {raw[k].max():+.4f}")

k = np.fft.fftfreq(D, d=apix)
s = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)


def lowpass(v, res_A, width_px=3.0):                   # raised cosine, as register_fraction.lowpass_filter
    ds = width_px / (D * apix)
    f = 0.5 - 0.5 * np.cos(np.pi * np.clip((1 / res_A + ds / 2 - s) / ds, 0, 1))
    return np.real(np.fft.ifftn(np.fft.fftn(v) * f))


mp = {n: lowpass(v, LOWPASS_A) for n, v in raw.items()}

# The swap as an array operation. cryoDRGN maps are indexed [z, y, x] with the origin at index D/2.
Pm = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
Carr = Pm @ C @ Pm.T


def swap(vol, c=np.zeros(3), order=1):                 # 2-fold about u through c (map-frame px from the origin)
    q0 = D / 2 + Pm @ np.asarray(c, float)
    return ndimage.affine_transform(vol, Carr, offset=q0 - Carr @ q0, order=order, mode="constant")


# Molecular mask as in register_fraction.py (15 A map, volume of MASK_KDA, largest blob, 6 A dilation), made
# symmetric under both swaps so that no comparison below is cut off by the mask.
V15 = lowpass(raw["V"], 15.0)
nvox = int(round(MASK_KDA * 1000 * 1.21 / apix ** 3))
lab, nlab = ndimage.label(V15 >= np.partition(V15.ravel(), -nvox)[-nvox])
blob = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(V15), lab, np.arange(1, nlab + 1))))
blob = ndimage.distance_transform_edt(~blob) * apix <= 6.0
env = blob | (swap(blob.astype(float), order=0) > 0.5) | (swap(blob.astype(float), cS, order=0) > 0.5)
vox = np.argwhere(env).astype(float)


def _cc(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")      # nan: one map is constant (e.g. M6 == M7)


def corr(a, b):
    return _cc(a[env], b[env])


def corr_sw(a, b, c=np.zeros(3)):                      # corr(a, swap(b, c)) in the mask, without a full resample
    q0 = D / 2 + Pm @ np.asarray(c, float)
    return _cc(a[env], ndimage.map_coordinates(b, ((vox - q0) @ Carr.T + q0).T, order=1, mode="constant"))


r_m7, r_m6 = corr(mp["M7"], mp["M7r"]), corr(mp["M6"], mp["M6r"])
r_conv = corr_sw(mp["M7C"], mp["M7r"])
r_FS, r_FSm = corr_sw(mp["F"], mp["M7r"], cS), corr_sw(mp["F"], mp["M7r"], -cS)
mp["S7"] = swap(mp["M7r"], cS)                         # M7 swapped about the shift-derived axis

names = ["V", "M7", "M7r", "M6", "M6r", "F", "M7C", "S7"]
print(f"\ncorrelation after a {LOWPASS_A:g} A low-pass, in a {MASK_KDA:g} kDa molecular mask ({env.sum():,} voxels):")
print("       " + "".join(f"{n:>8s}" for n in names))
for a in names:
    print(f"{a:>5s}  " + "".join(f"{corr(mp[a], mp[b]):8.4f}" for b in names))
print("M7r / M6r = the swap maps re-made from the current pose files; F = J487.C rotations + J486 shifts;\n"
      "M7C = J487.C rotations + J487 shifts; S7 = M7r swapped (array operation) about the shift-derived axis.")

# V's own best 2-fold axis position (u fixed, offset fitted), for comparison with the pose-derived axes.
best = min((optimize.minimize(lambda x: -corr_sw(mp["V"], mp["V"], Bp @ x), x0, method="Nelder-Mead",
                              options=dict(xatol=0.02, fatol=1e-6)) for x0 in (np.zeros(2), Bp.T @ cS)),
           key=lambda r: r.fun)
cV = Bp @ best.x
gap, gap_fr = np.linalg.norm(cV - cS) * apix, np.linalg.norm(cV - c_fr) * apix

r_6F, r_67, r_6S, r_6C = (corr(mp["M6r"], mp[n]) for n in ("F", "M7r", "S7", "M7C"))
r_6V = corr_sw(mp["M6r"], mp["M7r"], cV)               # what a check with V's own axis sees
reg = corr(mp["M6r"] - mp["M7r"], mp["S7"] - mp["M7r"])
anti = corr(mp["M6r"] - swap(mp["M6r"], cS), mp["M7r"] - mp["S7"])  # -1: M6 is M7 swapped; +1: same register
tol = max(0.005, 3 * (1 - r_conv))
ok_V = r_6S - r_6V <= tol                              # V's axis relates M6 to M7 as well as the shift-derived one
print("\nCHECKS")
print(f"  a. M7 reproducible   corr(M7, M7r)          = {r_m7:.4f}   (~1: run 1 used pose.pkl on this index file)")
print(f"  b. M6 reproducible   corr(M6, M6r)          = {r_m6:.4f}   (~1: run 2 used the current J486 pose file)")
print(f"  c. array convention  corr(M7C, swap(M7r))   = {r_conv:.4f}   (~1: the swap operator below is exact)")
print(f"  d. shift-derived axis corr(F, swap(M7r, c)) = {r_FS:.4f}   (opposite sign: {r_FSm:.4f}); axis "
      f"{np.linalg.norm(cS) * apix:.2f} +/- {se_c:.2f} A from the origin")
print(f"  e. M6r vs M7r        swapped about that axis {r_6S:.4f} | about the origin {r_6C:.4f} | "
      f"about V's own axis {r_6V:.4f} | unswapped {r_67:.4f}   (F: {r_6F:.4f})")
print(f"  f. register          corr(M6r - M7r, S7 - M7r) = {reg:+.3f} (+1: swapped);  antisymmetric parts "
      f"{anti:+.3f} (-1: swapped, +1: same register)")
print(f"  g. operator match    V's own 2-fold axis: {np.linalg.norm(cV) * apix:.2f} A from the origin, {gap:.2f} A "
      f"from the shift-derived axis" + (f" ({gap_fr:.2f} A after removing J486's frame offset)" if frame_A >= 0.5
      else "") + f";  corr(V, swap V) {-best.fun:.4f} there vs {corr_sw(mp['V'], mp['V'], cS):.4f}")
print(f"     map frames      J486 - J487 offset {frame_A:.2f} A (concordant particles)")

print("\nVERDICT")
prov_ok = r_m7 >= 0.999 and r_m6 >= 0.999
inv = "  A value near -1 would mean UNINVERT_DATA differs from that run." if min(r_m7, r_m6) < 0 else ""
print("  M7 on Drive " + ("is pose.pkl back-projected on this index file (reproduced)." if r_m7 >= 0.999 else
      f"is NOT pose.pkl back-projected on this index file (corr {r_m7:.4f}): run 1 used other inputs. "
      "Use J487_rerun as M7."))
print("  M6 on Drive " + ("is the current J486 pose file back-projected on this index file (reproduced)."
      if r_m6 >= 0.999 else f"is NOT the current J486 pose file back-projected on this index file (corr {r_m6:.4f}): "
      "run 2 used other inputs, or the pose file was rewritten since. Use J486_rerun as M6.") + inv)
pre = "  " if prov_ok else "  On the re-runs: "
swapped = r_6S > r_67 and r_FS >= r_FSm - 1e-3 and (reg > 0.5 or anti < -0.2)
if not r_conv >= 0.98:
    print("  The array convention check failed, so the operator comparisons are not interpretable. Send me the output.")
elif swapped:
    print(pre + f"M6 is M7 swapped about an axis {np.linalg.norm(cS) * apix:.2f} +/- {se_c:.2f} A from the box "
          f"origin, the axis the J486 - J487 shift changes imply"
          + ("." if reg > 0.5 else f" (reg {reg:+.2f} is low because M6r is blurrier than M7r; the antisymmetric "
                                   f"test, {anti:+.2f}, still shows the swap)."))
    if frame_A < 0.5:
        print(f"  J486 and J487 share a map frame (offset {frame_A:.2f} A), so this is the swap the two refinements "
              f"applied to these particles.")
    else:
        print(f"  J486's map frame is offset {frame_A:.2f} A from J487's. Without that offset the refinements' swap "
              f"axis is {gap_fr:.2f} A from V's own axis.")
    if ok_V:
        print(f"  V's own 2-fold axis ({gap:.2f} A away) also relates M6 to M7: swapping about it gives {r_6V:.4f}, "
              f"against {r_6S:.4f}.")
        if r_6S < 0.98:
            print(f"  But M6r is blurrier than M7r, so even the exact swap reaches only {r_6S:.4f}: check (e)'s 0.98 "
                  "bar cannot be met with these maps, whichever operator is used. Send me the output.")
        else:
            print("  " + ("So check (e) with V's operator should have passed on these maps. Send me the output."
                          if prov_ok else "So check (e) should pass on the re-runs; the Drive map that did not "
                          "reproduce is what failed it. Re-run register_fraction.py with J487_rerun and J486_rerun."))
    else:
        print(f"  V's own 2-fold axis is {gap:.2f} A from that one, and swapping M7 about it gives only {r_6V:.4f} "
              f"(against {r_6S:.4f}). A check that uses V's operator therefore fails, as check (e) did"
              + ("." if prov_ok else ", whatever the Drive map that did not reproduce contributes."))
        if frame_A < 0.5:
            print("  The refinements swapped these particles about the shift-derived axis, not V's, so a register "
                  "fraction estimated with V's axis (as register_fraction.py does) is biased. Send me this output and "
                  "I will set the estimate up with the shift-derived axis.")
        elif gap_fr <= 1.0:
            print("  Most of that gap is J486's frame offset: V's axis remains the one to use for p(M7), and M6 must "
                  "be translated by the frame offset before comparing it with M7. Send me this output.")
        else:
            print(f"  J486's frame offset explains only part of it: the refinements' swap axis is still {gap_fr:.2f} A "
                  "from V's. Send me this output.")
else:
    print(pre + "M6 is not M7 swapped about the shift-derived axis (see e and f). Send me the output.")
