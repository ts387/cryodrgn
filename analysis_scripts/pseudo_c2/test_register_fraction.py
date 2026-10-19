#!/usr/bin/env python3
"""
test_register_fraction.py -- synthetic validation of register_fraction.py

Builds a box-128 (2.609375 A/px) pseudo-C2 "dimer" (two similar Gaussian-blob protomers
related by a 2-fold about w, deliberately non-identical) plus an asymmetric "MPP6" blob on one
protomer.  From it the script makes cryoDRGN-like maps stored [z, y, x] (centre index 64):

    V  = (1-eps) T + eps C(T)                + noise          (reference register)
    M7 = a T + b S(T)      + s SYM           + noise
    M6 = b T + a S^-1(T)   + s SYM           + noise'         (noise' = C(noise) by default)

T is the true structure, S the swap operator (180 deg about w by default), and SYM a C-symmetric
contaminant: 'dimer' (symmetrised dimer without MPP6), 'junk' (rotational average of T), or
'mixed' (half of each).  The noise is Gaussian with a per-shell spectrum set by an SNR curve
(about FSC_half 0.143 at 5 A, 0.8 at 8 A, i.e. a 32k-particle backprojection), with
independent half maps.

Tests
  0  cryoDRGN convention: a numpy port of backproject_voxel (add_slice + ihtn_center).
     Is backproject(R7 S) = C_u(backproject(R7)) with u -> array [z,y,x] about index D/2?
  1  p recovered (+/-0.05) for (a,b,s) = (0.9,.05,.05), (.5,.5,0), (.3,.6,.1), (.6,0,.4);
     p6 ~ 1-p7; half-map spread and bootstrap SE vs Monte-Carlo truth.
  2  convention search and global 2-fold search: [z,y,x] storage, mirrored axes, np.flip
     mirror, off-centre 2-fold, 178 deg swap.
  3  robustness: low-pass 5/8/12 A, global vs focused mask, focus level, mask size, SYM type.
  4  V contaminated by 10% swapped particles: bias vs the analytic prediction.
  5  degenerate a = b (and a = b = 0).
  E  end-to-end CLI run on MRC files written with cryoDRGN-style headers (figures + report).

Run:  venv/bin/python test_register_fraction.py [outdir]
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import register_fraction as rf  # noqa: E402

APIX = 2.609375
D = 128
C0 = D / 2
U_REAL = np.array([0.720, -0.692, 0.061])
U_REAL = U_REAL / np.linalg.norm(U_REAL)
W_GEN = np.array([0.35, 0.55, -0.76])
W_GEN = W_GEN / np.linalg.norm(W_GEN)
P_ZYX = rf.parse_mapping("[+z,+y,+x]")
FAST = dict(global_search=False, n_boot=0, sweep_lowpass=(), sweep_focus=(), diagnostics=False, conv_offset=False)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "test_out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "test_report.txt"), "w")
RESULTS = {}


def log(s=""):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_axis_angle(w, deg):
    w = unit(w)
    t = math.radians(deg)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


# --------------------------------------------------------------------------------------
# Synthetic structure (map frame, Angstrom, relative to the box centre)
# --------------------------------------------------------------------------------------
class Structure:
    def __init__(self, w, seed=1, axis_point_A=(0.0, 0.0, 0.0)):
        rng = np.random.default_rng(seed)
        self.w = unit(w)
        self.ap = np.asarray(axis_point_A, float)
        e1, e2 = rf.tangent_basis(self.w)

        def ellipsoid(n, centre, semi):
            pts = []
            while len(pts) < n:
                q = rng.uniform(-1, 1, 3)
                if q @ q <= 1:
                    pts.append(q)
            q = np.array(pts) * semi
            return centre + q[:, :1] * e1 + q[:, 1:2] * e2 + q[:, 2:3] * self.w

        nA = 70
        cA = 17.0 * e1
        posA = ellipsoid(nA, cA, np.array([20.0, 14.0, 22.0]))
        ampA = rng.uniform(0.7, 1.3, nA)
        sigA = rng.uniform(2.6, 4.0, nA)
        R2 = rot_axis_angle(self.w, 180)
        posB = posA @ R2.T + rng.normal(0, 1.0, posA.shape)  # pseudo-symmetric copy
        ampB = ampA * (1 + rng.normal(0, 0.15, nA))
        keep = np.ones(nA, bool)
        keep[rng.choice(nA, 5, replace=False)] = False
        extra = ellipsoid(4, cA @ R2.T, np.array([18.0, 12.0, 20.0]))
        posB = np.vstack([posB[keep], extra])
        ampB = np.concatenate([ampB[keep], rng.uniform(0.7, 1.3, 4)])
        sigB = np.concatenate([sigA[keep], rng.uniform(2.6, 4.0, 4)])
        nM = 18  # MPP6 ~13% of the mass, on protomer A, away from the 2-fold
        cM = cA + 8.0 * e1 + 24.0 * e2 + 10.0 * self.w
        posM = ellipsoid(nM, cM, np.array([11.0, 9.0, 9.0]))
        self.parts = {"A": (posA, ampA, sigA), "B": (posB, ampB, sigB),
                      "M": (posM, rng.uniform(0.7, 1.3, nM), rng.uniform(2.6, 4.0, nM))}
        for k in self.parts:  # move the whole thing so the 2-fold passes through ap
            p, a, s = self.parts[k]
            self.parts[k] = (p + self.ap, a, s)

    def blobs(self, parts=("A", "B", "M"), Q=None):
        """Concatenate parts, optionally transformed by rotation Q about the axis point."""
        P = np.vstack([self.parts[k][0] for k in parts])
        a = np.concatenate([self.parts[k][1] for k in parts])
        s = np.concatenate([self.parts[k][2] for k in parts])
        if Q is not None:
            P = (P - self.ap) @ Q.T + self.ap
        return P, a, s


def render(blobs, P_map2arr=P_ZYX, centre=C0, box=D, apix=APIX):
    """Sum of Gaussians; map-frame (x,y,z) A -> array index = centre + P (r / apix)."""
    pos, amp, sig = blobs
    vol = np.zeros((box,) * 3, np.float32)
    cen = np.broadcast_to(np.asarray(centre, float), (3,))
    for r, a, s in zip(pos, amp, sig):
        x = cen + P_map2arr @ (r / apix)
        sp = s / apix
        h = int(math.ceil(4 * sp))
        lo = np.floor(x).astype(int) - h
        hi = lo + 2 * h + 2
        lo_c, hi_c = np.maximum(lo, 0), np.minimum(hi, box)
        g = [np.exp(-((np.arange(lo_c[i], hi_c[i]) - x[i]) ** 2) / (2 * sp * sp)) for i in range(3)]
        vol[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]] += a * (
            g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :])
    return vol


def radial_average(vol, centre=C0):
    r = np.sqrt(((np.indices(vol.shape) - centre) ** 2).sum(0))
    rb = np.round(r * 2).astype(int)
    prof = np.bincount(rb.ravel(), vol.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    return prof[rb].astype(np.float32)


# --------------------------------------------------------------------------------------
# Noise with a realistic per-shell SNR
# --------------------------------------------------------------------------------------
def snr_full(s):
    """SNR of a 32k-particle full map vs spatial frequency s (1/A): ~20 at 10 A, ~9 at 8 A, ~0.33 at 5 A."""
    return 20.0 * np.exp(-137.0 * (s ** 2 - 0.01))


class NoiseGen:
    def __init__(self, signal, noise_scale=1.0):
        sh = signal.shape
        s = rf.freq_radius(sh, APIX)
        self.rsh = np.round(s * sh[0] * APIX).astype(int)
        h = sfft.rfftn(signal, workers=-1)
        pw = np.bincount(self.rsh.ravel(), (np.abs(h) ** 2).ravel()) / np.maximum(np.bincount(self.rsh.ravel()), 1)
        smid = np.arange(len(pw)) / (sh[0] * APIX)
        npow = pw / snr_full(smid) * noise_scale
        npow[0] = npow[1]
        self.gain = np.sqrt(npow / signal.size)[self.rsh].astype(np.float32)
        self.shape = sh

    def draw(self, rng, var_factor=1.0):
        W = rng.standard_normal(self.shape, dtype=np.float32)
        h = sfft.rfftn(W, workers=-1) * self.gain * math.sqrt(var_factor)
        return sfft.irfftn(h, s=self.shape, workers=-1).astype(np.float32)


def fsc(a, b, apix=APIX):
    ha, hb = sfft.rfftn(a), sfft.rfftn(b)
    rsh = np.round(rf.freq_radius(a.shape, apix) * a.shape[0] * apix).astype(int)
    num = np.bincount(rsh.ravel(), (ha * np.conj(hb)).real.ravel())
    den = np.sqrt(np.bincount(rsh.ravel(), (np.abs(ha) ** 2).ravel()) * np.bincount(rsh.ravel(), (np.abs(hb) ** 2).ravel()))
    return np.arange(len(num)) / (a.shape[0] * apix), num / np.maximum(den, 1e-30)


# --------------------------------------------------------------------------------------
# Dataset builder
# --------------------------------------------------------------------------------------
_CACHE = {}


def arr_rotation(Q_map, P, centre, ap_A):
    """Volume transform for a map-frame rotation Q about map point ap_A: returns f(vol) = vol o g^-1."""
    Qa = P @ Q_map @ P.T
    ca = np.asarray(centre, float) + P @ (np.asarray(ap_A, float) / APIX)
    M = Qa.T
    off = ca - M @ ca

    def f(vol):
        return ndimage.affine_transform(vol, M, offset=off, order=3, mode="grid-wrap").astype(np.float32)

    return f


def truth_maps(w=U_REAL, P=P_ZYX, centre=C0, ap_A=(0, 0, 0), seed=1, rot_deg=180.0):
    key = (tuple(np.round(w, 6)), tuple(P.ravel()), np.asarray(centre, float).tobytes(), tuple(ap_A), seed, rot_deg)
    if key in _CACHE:
        return _CACHE[key]
    st = Structure(w, seed=seed, axis_point_A=ap_A)
    Qc = rot_axis_angle(st.w, 180)
    Qp, Qm = rot_axis_angle(st.w, rot_deg), rot_axis_angle(st.w, -rot_deg)
    T = render(st.blobs(), P, centre)
    out = dict(
        T=T,
        CT=render(st.blobs(Q=Qc), P, centre),
        ST=render(st.blobs(Q=Qp), P, centre),
        SiT=render(st.blobs(Q=Qm), P, centre),
        dimer=0.5 * (render(st.blobs(("A", "B")), P, centre) + render(st.blobs(("A", "B"), Q=Qc), P, centre)),
        rotC=arr_rotation(Qc, P, centre, st.ap),
        axis_array=unit(P @ st.w),
        structure=st,
    )
    out["junk"] = radial_average(T, centre)
    out["noise"] = NoiseGen(T)
    _CACHE[key] = out
    return out


def sym_component(tm, sym):
    if sym == "none":
        return np.zeros_like(tm["T"])
    if sym == "dimer":
        return tm["dimer"]
    if sym == "junk":
        return tm["junk"]
    if sym == "mixed":
        return 0.5 * tm["dimer"] + 0.5 * tm["junk"]
    raise ValueError(sym)


def build(a, b, s, sym="mixed", eps=0.0, noise_scale=1.0, m6_noise="rotated", seed=0, **tkw):
    tm = truth_maps(**tkw)
    rng = np.random.default_rng(seed)
    S = sym_component(tm, sym)
    sigV = (1 - eps) * tm["T"] + eps * tm["CT"]
    sig7 = a * tm["T"] + b * tm["ST"] + s * S
    sig6 = b * tm["T"] + a * tm["SiT"] + s * S
    ng = tm["noise"]

    def halves(sig, n=None):
        if n is None:
            n = (ng.draw(rng, 2 * noise_scale), ng.draw(rng, 2 * noise_scale))
        A_, B_ = sig + n[0], sig + n[1]
        return dict(full=(A_ + B_) / 2, a=A_, b=B_), n

    V, _ = halves(sigV)
    M7, n7 = halves(sig7)
    if m6_noise == "rotated":
        M6, _ = halves(sig6, (tm["rotC"](n7[0]), tm["rotC"](n7[1])))
    else:
        M6, _ = halves(sig6)
    return dict(V=V, M7=M7, M6=M6), tm


def p_true(a, b):
    return a / (a + b) if a + b > 0 else float("nan")


def run(maps, u_map, **opts):
    o = dict(FAST)
    o.update(opts)
    return rf.run_analysis(maps, APIX, u_map, o, log=lambda s: None)


# --------------------------------------------------------------------------------------
# Test 0: cryoDRGN backprojection convention (numpy port of backproject_voxel)
# --------------------------------------------------------------------------------------
def random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def htn_center(v):
    f = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(v)))
    return f.real - f.imag


def ihtn_center(v):
    f = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(v))) / v.size
    return f.real - f.imag


def cryodrgn_backproject(ff, rots, Dl, reg_weight=1.0, batch=300):
    """numpy port of cryodrgn 4.3.1 backproject_voxel: add_slice + regularize_volume.
    ff: (N, npix) Hartley image values on the lattice mask; rots: (N,3,3); Dl = lattice size (odd)."""
    d2 = Dl // 2
    lin = np.linspace(-d2, d2, Dl)
    x0, x1 = np.meshgrid(lin, lin)
    coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(Dl * Dl)], 1)
    lmask = (coords[:, 0] ** 2 + coords[:, 1] ** 2) <= d2 ** 2
    coords = coords[lmask]
    vol = np.zeros(Dl ** 3)
    cnt = np.zeros(Dl ** 3)
    for s in range(0, len(rots), batch):
        rb = rots[s:s + batch]
        fb = ff[s:s + batch]
        fc = np.clip(coords[None] @ rb, -d2, d2)  # ff_coord = lattice.coords @ rot  -> (B, P, 3) = (x, y, z)
        fl, ce = np.floor(fc).astype(int), np.ceil(fc).astype(int)
        for cx in (fl[..., 0], ce[..., 0]):
            for cy in (fl[..., 1], ce[..., 1]):
                for cz in (fl[..., 2], ce[..., 2]):
                    dist = np.stack([cx, cy, cz], -1) - fc
                    wv = 1 - np.sqrt((dist ** 2).sum(-1))
                    wv[wv < 0] = 0
                    flat = ((cz + d2) * Dl ** 2 + (cy + d2) * Dl + (cx + d2)).ravel()
                    vol += np.bincount(flat, (wv * fb).ravel(), minlength=Dl ** 3)
                    cnt += np.bincount(flat, wv.ravel(), minlength=Dl ** 3)
    # (the 8 corners in cryodrgn are the 8 floor/ceil combinations - same set as above)
    vol = vol.reshape(Dl, Dl, Dl)
    cnt = cnt.reshape(Dl, Dl, Dl)
    cnt[cnt == 0] = 1
    rc = cnt + reg_weight * cnt.mean()
    rc *= cnt.mean() / rc.mean()
    return ihtn_center((vol / rc)[:-1, :-1, :-1])


def test0():
    log("=" * 100)
    log("TEST 0  cryoDRGN backproject_voxel convention (numpy port of add_slice + ihtn_center), box 64")
    t0 = time.time()
    rng = np.random.default_rng(7)
    D0, apix0 = 64, APIX * 2
    st = Structure(W_GEN, seed=3)
    vol = render(st.blobs(), P_ZYX, D0 / 2, D0, apix0)  # stored [z,y,x], centre 32
    Dl = D0 + 1
    ht = htn_center(vol)
    ht = np.pad(ht, ((0, 1),) * 3, mode="wrap")  # symmetrize_ht analogue: index 64 == index 0
    d2 = Dl // 2
    lin = np.linspace(-d2, d2, Dl)
    x0, x1 = np.meshgrid(lin, lin)
    coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(Dl * Dl)], 1)
    coords = coords[(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= d2 ** 2]
    N = 1500
    R7 = random_rotations(N, rng)
    fc = coords[None] @ R7  # (N, P, 3) map-frame (x, y, z) Fourier coords for each image pixel
    # forward model = cryoDRGN's: image Hartley value = volume HT at coords @ rot  -> array [z, y, x]
    ff = ndimage.map_coordinates(ht, [fc[..., 2].ravel() + d2, fc[..., 1].ravel() + d2, fc[..., 0].ravel() + d2],
                                 order=1).reshape(N, -1)
    res = {}
    for name, u in (("generic u", unit([0.25, -0.6, 0.76])), ("real u", U_REAL)):
        S = rot_axis_angle(u, 180)
        B7 = cryodrgn_backproject(ff, R7, Dl)
        B6 = cryodrgn_backproject(ff, R7 @ S, Dl)  # R6 = R7 S  <=>  R7^T R6 = S (axis u)
        # orientation: is B7 the input volume itself (no mirror / transpose)?
        m = vol > np.percentile(vol, 90)
        orient = {"identity": rf.wcorr(B7[m], vol[m], np.ones(m.sum())),
                  "flip axis2(x)": rf.wcorr(B7[m], np.roll(vol[:, :, ::-1], 1, 2)[m], np.ones(m.sum())),
                  "transpose z<->x": rf.wcorr(B7[m], vol.transpose(2, 1, 0)[m], np.ones(m.sum()))}
        # which mapping of u turns B7 into B6?
        sp = rf.Spectra(B7.shape, apix0)
        _, c7 = sp.lowpass("B7", B7, 12.0, coeffs=True)
        b6 = sp.lowpass("B6", B6, 12.0)
        X = np.argwhere(m).astype(float)
        rows = []
        for lab, alt, P in rf.mapping_classes():
            for cen in (D0 / 2, D0 / 2 - 0.5):
                op = rf.TwoFold(P @ u, cen)
                rows.append((rf.wcorr(b6[m], op.sample(c7, X), np.ones(len(X))), lab, cen))
        rows.sort(reverse=True)
        res[name] = dict(orientation=orient, best=rows[0], second=rows[1],
                         hyp=[r for r in rows if r[1] == "[+z,+y,+x]" and r[2] == D0 / 2][0])
        log(f"  {name}: corr(backproject(R7), input volume) identity {orient['identity']:.3f} | "
            f"x-flipped {orient['flip axis2(x)']:.3f} | z<->x transposed {orient['transpose z<->x']:.3f}")
        log(f"  {name}: backproject(R7 S) vs C_u(backproject(R7)): best mapping {rows[0][1]} centre {rows[0][2]} "
            f"corr {rows[0][0]:.4f}; 2nd {rows[1][1]} centre {rows[1][2]} corr {rows[1][0]:.4f}")
    ok = all(r["best"][1] == "[+z,+y,+x]" and r["best"][2] == D0 / 2 and r["orientation"]["identity"] > 0.9
             for r in res.values())
    log(f"  PASS={ok}  ({time.time() - t0:.1f}s)  -> u (axis of R7^T R6, map xyz) maps to array [z,y,x], "
        f"centre D/2, no mirror.")
    RESULTS["test0"] = dict(pass_=ok, detail={k: dict(best=v["best"], second=v["second"], hyp=v["hyp"],
                                                       orientation=v["orientation"]) for k, v in res.items()})
    return ok


# --------------------------------------------------------------------------------------
# Test 1: recovery of p
# --------------------------------------------------------------------------------------
SCEN = [(0.9, 0.05, 0.05), (0.5, 0.5, 0.0), (0.3, 0.6, 0.1), (0.6, 0.0, 0.4)]


def test1():
    log("=" * 100)
    log("TEST 1  recovery of p = a/(a+b); SYM = 'mixed' (half symmetric dimer w/o MPP6, half junk); "
        "M6 noise = C(M7 noise)")
    t0 = time.time()
    tm = truth_maps()
    maps, _ = build(0.3, 0.6, 0.1, seed=11)
    s, f = fsc(maps["V"]["a"], maps["V"]["b"])
    i143 = np.argmax(f[1:] < 0.143) + 1
    log(f"  noise check: V half-map FSC 0.143 at {1 / s[i143]:.2f} A; FSC at 8 A = {np.interp(1 / 8, s, f):.3f}; "
        f"at 12 A = {np.interp(1 / 12, s, f):.3f}")
    rows, allok = [], True
    log(f"  {'a,b,s':>16s} {'p_true':>7s} | {'focused p7':>10s} {'1-p6':>6s} {'p7+p6':>6s} {'SEhalf':>7s} {'SEboot':>7s} "
        f"{'a+b fit':>7s} | {'global p7':>9s} {'1-p6':>6s} | pass(focused)")
    for i, (a, b, s_) in enumerate(SCEN):
        maps, _ = build(a, b, s_, seed=100 + i)
        R, _ = run(maps, U_REAL, n_boot=30, seed=i)
        fo, gl = R["fits"]["focused"], R["fits"]["global"]
        pt = p_true(a, b)
        ok = abs(fo["M7"]["p"] - pt) <= 0.05 and abs((1 - fo["M6"]["p"]) - pt) <= 0.05
        allok &= ok
        rows.append(dict(a=a, b=b, s=s_, p_true=pt, p7_focused=fo["M7"]["p"], p6_focused=fo["M6"]["p"],
                         se_half=fo["M7"]["se_half"], se_boot=fo["M7"]["se_boot"], sum_focused=fo["M7"]["sum"],
                         p7_global=gl["M7"]["p"], p6_global=gl["M6"]["p"], antisym_p7=fo["M7"]["antisym"]["p"],
                         halfM=fo["M7"]["p_halfM"], halfV=fo["M7"]["p_halfV"], pass_=ok,
                         mapping=R["convention"]["best"]["label"]))
        log(f"  {str((a, b, s_)):>16s} {pt:7.3f} | {fo['M7']['p']:10.3f} {1 - fo['M6']['p']:6.3f} "
            f"{fo['M7']['p'] + fo['M6']['p']:6.3f} {fo['M7']['se_half']:7.3f} {fo['M7']['se_boot']:7.3f} "
            f"{fo['M7']['sum']:7.3f} | {gl['M7']['p']:9.3f} {1 - gl['M6']['p']:6.3f} | {ok}")
    # same scenarios, SYM orthogonal-ish ('none'): global mask must also work when the model holds
    log("  control: SYM = 'none' (model exactly as assumed) -> global mask should also be unbiased")
    for i, (a, b, s_) in enumerate(SCEN):
        maps, _ = build(a, b, s_, sym="none", seed=200 + i)
        R, _ = run(maps, U_REAL)
        log(f"    {str((a, b, s_)):>16s} p_true {p_true(a, b):.3f}: focused {R['fits']['focused']['M7']['p']:.3f}  "
            f"global {R['fits']['global']['M7']['p']:.3f}")
        rows.append(dict(a=a, b=b, s=s_, sym="none", p7_focused=R['fits']['focused']['M7']['p'],
                         p7_global=R['fits']['global']['M7']['p']))
    # independent M6 noise: p7 + p6 is then no longer 1 by construction
    maps, _ = build(0.3, 0.6, 0.1, m6_noise="independent", seed=300)
    R, _ = run(maps, U_REAL, diagnostics=True)
    maps2, _ = build(0.3, 0.6, 0.1, m6_noise="rotated", seed=300)
    R2, _ = run(maps2, U_REAL, diagnostics=True)
    p76_ind, c_ind = R["summary"]["p7_plus_p6"], R["diagnostics"]["corr_M6_CM7"]
    log(f"  M6 noise = C(M7 noise): p7+p6 = {R2['summary']['p7_plus_p6']:.4f}, corr(M6, C M7) = "
        f"{R2['diagnostics']['corr_M6_CM7']:.4f};  independent M6 noise: p7+p6 = {R['summary']['p7_plus_p6']:.4f}, "
        f"corr(M6, C M7) = {R['diagnostics']['corr_M6_CM7']:.4f}")
    # Monte-Carlo calibration of the uncertainty estimates
    log("  uncertainty calibration: (a,b,s) = (0.3,0.6,0.1), 8 independent noise realisations")
    ps, seh, seb, psg = [], [], [], []
    for k in range(8):
        maps, _ = build(0.3, 0.6, 0.1, seed=1000 + k)
        R, _ = run(maps, U_REAL, n_boot=15, seed=k)
        ps.append(R["fits"]["focused"]["M7"]["p"])
        psg.append(R["fits"]["global"]["M7"]["p"])
        seh.append(R["fits"]["focused"]["M7"]["se_half"])
        seb.append(R["fits"]["focused"]["M7"]["se_boot"])
    cal = dict(mc_sd_focused=float(np.std(ps, ddof=1)), mc_mean_focused=float(np.mean(ps)),
               rms_se_half=float(np.sqrt(np.mean(np.square(seh)))), mean_se_boot=float(np.mean(seb)),
               mc_sd_global=float(np.std(psg, ddof=1)), mc_mean_global=float(np.mean(psg)),
               se_half_range=[float(np.min(seh)), float(np.max(seh))])
    log(f"    focused: MC mean p {cal['mc_mean_focused']:.4f} (true 0.3333), MC SD {cal['mc_sd_focused']:.4f}; "
        f"RMS SE_half {cal['rms_se_half']:.4f} (range {cal['se_half_range'][0]:.4f}-{cal['se_half_range'][1]:.4f}); "
        f"mean SE_boot {cal['mean_se_boot']:.4f}")
    log(f"    global : MC mean p {cal['mc_mean_global']:.4f}, MC SD {cal['mc_sd_global']:.4f}")
    # noisier data (x4 noise power ~ 8k particles)
    maps, _ = build(0.3, 0.6, 0.1, noise_scale=4.0, seed=400)
    R, _ = run(maps, U_REAL, n_boot=30)
    log(f"  noise x4 (~8k particles): focused p7 {R['fits']['focused']['M7']['p']:.3f} "
        f"SE_boot {R['fits']['focused']['M7']['se_boot']:.3f}; global p7 {R['fits']['global']['M7']['p']:.3f}")
    log(f"  PASS={allok}  ({time.time() - t0:.1f}s)")
    RESULTS["test1"] = dict(pass_=allok, rows=rows, calibration=cal,
                            p7_plus_p6=dict(rotated=R2["summary"]["p7_plus_p6"], independent=p76_ind),
                            corr_M6_CM7=dict(rotated=R2["diagnostics"]["corr_M6_CM7"], independent=c_ind))
    return allok


# --------------------------------------------------------------------------------------
# Test 2: conventions
# --------------------------------------------------------------------------------------
def test2():
    log("=" * 100)
    log("TEST 2  axis-convention search + independent global 2-fold search; (a,b,s) = (0.6,0.3,0.1), SYM mixed")
    t0 = time.time()
    cases = [
        ("u_real, [z,y,x]", U_REAL, "[+z,+y,+x]", {}),
        ("w generic, [z,y,x]", W_GEN, "[+z,+y,+x]", {}),
        ("w generic, x mirrored", W_GEN, "[+z,+y,-x]", {}),
        ("u_real, x mirrored", U_REAL, "[+z,+y,-x]", {}),
        ("w generic, z mirrored", W_GEN, "[-z,+y,+x]", {}),
        ("u_real, z mirrored", U_REAL, "[-z,+y,+x]", {}),
        ("w generic, axes permuted [x,z,y]", W_GEN, "[+x,+z,+y]", {}),
    ]
    rows, allok = [], True
    for name, w, lab, extra in cases:
        P = rf.parse_mapping(lab)
        maps, tm = build(0.6, 0.3, 0.1, w=w, P=P, seed=500)
        R, A = run(maps, w, global_search=True)
        truth = tm["axis_array"]
        best = R["convention"]["best"]
        ang_best = rf.line_angle(best["axis_array"], truth)
        ang_glob = rf.line_angle(R["operator_used"]["axis_array"], truth)
        exp_canon = lab if lab[1] == "+" else rf.mapping_classes()[0][0]  # placeholder, replaced below
        # canonical label of the true class (first sign +)
        Pc = P if P[0][np.nonzero(P[0])[0][0]] > 0 else -P
        canon = [c[0] for c in rf.mapping_classes() if np.allclose(c[2], Pc)][0]
        label_ok = best["label"] == canon
        # the best self-overlap 2-fold of the NOISE-FREE structure is already 2.3-2.6 deg from the
        # generating axis (protomers deliberately non-identical), so "agrees" means < 5 deg here
        ok = ang_best < 3.0 and ang_glob < 5.0 and best["centre"] == C0
        allok &= ok
        sc = R["selfcorr"]
        rows.append(dict(case=name, true_class=canon, chosen=best["label"], chosen_centre=best["centre"],
                         label_exact=label_ok, angle_chosen_deg=ang_best, angle_global_deg=ang_glob,
                         contrast=R["convention"]["contrast_vs_next_distinct"],
                         next_distinct=R["convention"]["next_distinct"]["label"],
                         corr_best=best["corr"], corr_refined=sc.get("refined"),
                         second=R["convention"]["table"][1]["label"],
                         second_angle=rf.line_angle(R["convention"]["table"][1]["axis_array"], truth),
                         second_corr=R["convention"]["table"][1]["corr"],
                         p7=R["fits"]["focused"]["M7"]["p"], pass_=ok))
        log(f"  {name:34s} true {canon:12s} chosen {best['label']:12s}@{best['centre']:.1f} "
            f"({ang_best:4.1f} deg) exact={label_ok!s:5s} | runner-up {R['convention']['table'][1]['label']} "
            f"{rows[-1]['second_angle']:.1f} deg, dcorr {best['corr'] - rows[-1]['second_corr']:.4f} | "
            f"next >15deg: dcorr {rows[-1]['contrast']:.3f} | global {ang_glob:4.2f} deg | p7 {rows[-1]['p7']:.3f} | {ok}")
    # np.flip mirror: centre moves to 63 along the flipped axis
    maps, tm = build(0.6, 0.3, 0.1, w=W_GEN, seed=501)
    flipped = {k: {h: np.ascontiguousarray(v[h][:, :, ::-1]) for h in v} for k, v in maps.items()}
    R, _ = run(flipped, W_GEN, global_search=True)
    sc = R["selfcorr"]
    log(f"  np.flip(axis 2) mirror (centre -> 63 on that axis): chosen {R['convention']['best']['label']} "
        f"@{R['convention']['best']['centre']}, refined offset {R['operator_used']['axis_offset_A']} A, "
        f"selfcorr nominal {sc['nominal']:.4f} -> refined {sc['refined']:.4f}; "
        f"p7 refined {R['fits']['focused']['M7']['p']:.3f} vs nominal "
        f"{R['operator_sensitivity']['nominal']['focused']['M7']:.3f} (true 0.667)")
    rows.append(dict(case="np.flip axis2", chosen=R["convention"]["best"]["label"],
                     offset_A=R["operator_used"]["axis_offset_A"], p7_refined=R["fits"]["focused"]["M7"]["p"],
                     p7_nominal=R["operator_sensitivity"]["nominal"]["focused"]["M7"]))
    # off-centre pseudo-2-fold (axis passes 8 A from the box centre); swap operator is the same off-centre 2-fold
    e1, _ = rf.tangent_basis(U_REAL)
    ap = tuple((8.0 * e1).tolist())
    maps, tm = build(0.6, 0.3, 0.1, w=U_REAL, ap_A=ap, seed=502)
    R, _ = run(maps, U_REAL, global_search=True)
    os_ = R["operator_sensitivity"]
    log(f"  off-centre 2-fold (8.0 A): refined offset {R['operator_used']['axis_offset_A']} A; p7 focused: "
        f"refined {R['fits']['focused']['M7']['p']:.3f} | nominal+offset {os_['nominal+offset']['focused']['M7']:.3f} | "
        f"nominal (through centre) {os_['nominal']['focused']['M7']:.3f}  (true 0.667); selfcorr "
        f"{ {k: round(v, 3) for k, v in R['selfcorr'].items()} }")
    rows.append(dict(case="off-centre axis 8 A", offset_found_A=R["operator_used"]["axis_offset_A"],
                     p7_refined=R["fits"]["focused"]["M7"]["p"], p7_nominal=os_["nominal"]["focused"]["M7"],
                     p7_nominal_offset=os_["nominal+offset"]["focused"]["M7"]))
    ok_off = abs(R["fits"]["focused"]["M7"]["p"] - 2 / 3) <= 0.05 and abs(R["operator_used"]["axis_offset_A"] - 8.0) < 1.0
    # 178 deg swap
    maps, tm = build(0.6, 0.3, 0.1, w=U_REAL, rot_deg=178.0, seed=503)
    R, _ = run(maps, U_REAL, global_search=True)
    log(f"  swap = 178 deg (not an exact 2-fold): p7 focused {R['fits']['focused']['M7']['p']:.3f}, "
        f"1-p6 {1 - R['fits']['focused']['M6']['p']:.3f}, global {R['fits']['global']['M7']['p']:.3f} (true 0.667)")
    rows.append(dict(case="178 deg swap", p7=R["fits"]["focused"]["M7"]["p"], p6=R["fits"]["focused"]["M6"]["p"]))
    allok &= ok_off
    log(f"  PASS={allok}  ({time.time() - t0:.1f}s)")
    RESULTS["test2"] = dict(pass_=allok, rows=rows)
    return allok


# --------------------------------------------------------------------------------------
# Test 3: robustness to low-pass, mask, SYM type
# --------------------------------------------------------------------------------------
def test3():
    log("=" * 100)
    log("TEST 3  robustness: p(M7) vs low-pass (5/8/12 A) x mask (global / focused default level .3 excl .5),")
    log("        focused-mask variants (level, excl), envelope size (kDa), SYM type")
    t0 = time.time()
    rows = []
    combos = [((0.9, 0.05, 0.05), "mixed"), ((0.3, 0.6, 0.1), "mixed"), ((0.6, 0.0, 0.4), "none"),
              ((0.6, 0.0, 0.4), "junk"), ((0.6, 0.0, 0.4), "dimer"), ((0.6, 0.0, 0.4), "mixed")]
    fvar = ((0.3, 0.0), (0.5, 0.0), (0.2, 0.5), (0.3, 0.5), (0.3, 0.7))
    log(f"  {'(a,b,s) SYM':>26s} p_true | " + " ".join(f"{m[:4]}@{lp:<4g}" for m in ("glob", "foc") for lp in (5, 8, 12))
        + " | focused (level,excl) " + " ".join(f"{l:.1f}/{e:.1f}" for l, e in fvar) + " | kda 90/125/180")
    for i, ((a, b, s_), sym) in enumerate(combos):
        maps, _ = build(a, b, s_, sym=sym, seed=600 + i)
        R, _ = run(maps, U_REAL, sweep_lowpass=(5.0, 8.0, 12.0), sweep_focus=fvar)
        sw = R["sweep"]
        g = {(x["mask"], x["lowpass"]): x["p"] for x in sw if x["kind"] == "lowpass" and x["map"] == "M7"}
        fl = {(x["focus_level"], x["focus_excl"]): x["p"] for x in sw if x["kind"] == "focus" and x["map"] == "M7"}
        kd = {}
        for kda in (90.0, 125.0, 180.0):
            Rk, _ = run(maps, U_REAL, mask_kda=kda)
            kd[kda] = Rk["fits"]["focused"]["M7"]["p"]
        pt = p_true(a, b)
        log(f"  {str((a, b, s_)) + ' ' + sym:>26s} {pt:6.3f} | "
            + " ".join(f"{g[(m, lp)]:9.3f}" for m in ("global", "focused") for lp in (5.0, 8.0, 12.0))
            + " |                      " + " ".join(f"{fl.get(k, float('nan')):7.3f}" for k in fvar)
            + " | " + "/".join(f"{kd[k]:.3f}" for k in (90.0, 125.0, 180.0)))
        rows.append(dict(a=a, b=b, s=s_, sym=sym, p_true=pt, lowpass_mask={f"{m}@{lp}": g[(m, lp)] for (m, lp) in g},
                         focus_variants={f"{l}/{e}": v for (l, e), v in fl.items()}, mask_kda=kd,
                         alpha_plus_beta=R["fits"]["focused"]["M7"]["sum"]))
    worst = {}
    for r in rows:
        for k, v in list(r["lowpass_mask"].items()) + [("focused level/excl " + k, v) for k, v in r["focus_variants"].items()] \
                + [(f"kda {k}", v) for k, v in r["mask_kda"].items()]:
            worst[k] = max(worst.get(k, 0), abs(v - r["p_true"]))
    log("  worst |p - p_true| over the 6 combos:")
    for k, v in worst.items():
        log(f"     {k:32s} {v:.3f}")
    ok = worst["focused@8.0"] <= 0.05
    log(f"  PASS(default focused@8A within 0.05 for all combos)={ok}  ({time.time() - t0:.1f}s)")
    RESULTS["test3"] = dict(pass_=ok, rows=rows, worst=worst)
    return ok


# --------------------------------------------------------------------------------------
# Test 4: contaminated reference
# --------------------------------------------------------------------------------------
def test4():
    log("=" * 100)
    log("TEST 4  V itself contaminated by eps = 10% swapped particles (V = 0.9 T + 0.1 C T)")
    t0 = time.time()
    eps = 0.1
    rows = []
    for i, (a, b, s_) in enumerate([(0.9, 0.05, 0.05), (0.6, 0.3, 0.1), (0.3, 0.6, 0.1), (0.6, 0.0, 0.4)]):
        maps, _ = build(a, b, s_, eps=eps, seed=700 + i)
        R, _ = run(maps, U_REAL)
        pt = p_true(a, b)
        # With the third (C-symmetric) regressor, contamination eps acts like eps / r with
        # r = (alpha+beta)/kappa (the 2-regressor fit is the r = 1 case).
        r = R["summary"]["alpha_plus_beta_rel_M7"]
        pred = 0.5 + (pt - 0.5) / (1 - 2 * eps / r)
        pf, pg = R["fits"]["focused"]["M7"]["p"], R["fits"]["global"]["M7"]["p"]
        corr = 0.5 + (pf - 0.5) * (1 - 2 * eps / r)
        rows.append(dict(a=a, b=b, s=s_, p_true=pt, p_focused=pf, p_global=pg, predicted=pred, corrected=corr,
                         beta=R["fits"]["focused"]["M7"]["beta"]))
        log(f"  (a,b,s)={str((a, b, s_)):16s} p_true {pt:.3f} -> focused {pf:.3f} (bias {pf - pt:+.3f}; predicted "
            f"0.5+(p-0.5)/(1-2eps/r) = {pred:.3f}); global {pg:.3f}; beta {rows[-1]['beta']:+.3f}; "
            f"de-biased with known eps {corr:.3f}")
    ok = all(abs(r["p_focused"] - r["predicted"]) <= 0.05 for r in rows)
    log(f"  PASS(bias matches prediction within 0.05)={ok}  ({time.time() - t0:.1f}s)")
    RESULTS["test4"] = dict(pass_=ok, rows=rows)
    return ok


# --------------------------------------------------------------------------------------
# Test 5: degenerate a = b
# --------------------------------------------------------------------------------------
def test5():
    log("=" * 100)
    log("TEST 5  degenerate cases a = b (incl. a = b = 0: every swapped particle symmetric / register-less)")
    t0 = time.time()
    rows = []
    for i, (a, b, s_, sym) in enumerate([(0.5, 0.5, 0.0, "none"), (0.3, 0.3, 0.4, "mixed"),
                                         (0.05, 0.05, 0.9, "dimer"), (0.0, 0.0, 1.0, "dimer"),
                                         (0.0, 0.0, 1.0, "mixed")]):
        maps, _ = build(a, b, s_, sym=sym, seed=800 + i)
        warn = []
        o = dict(FAST)
        o.update(n_boot=40, seed=i)
        R, _ = rf.run_analysis(maps, APIX, U_REAL, o, log=lambda s: warn.append(s) if "WARNING" in s else None)
        f = R["fits"]["focused"]["M7"]
        g = R["fits"]["global"]["M7"]
        rows.append(dict(a=a, b=b, s=s_, sym=sym, p=f["p"], alpha=f["alpha"], beta=f["beta"], sum=f["sum"],
                         diff=f["diff"], se_half=f["se_half"], se_boot=f["se_boot"], boot16_84=f["boot_p16_p84"],
                         frac_outside=f["boot_frac_outside_0_1"], p_global=g["p"], se_boot_global=g["se_boot"],
                         sum_global=g["sum"], warned=bool(warn)))
        log(f"  (a,b,s)={str((a, b, s_)):17s} {sym:5s}: focused p {f['p']:7.3f} (alpha {f['alpha']:+.3f}, beta {f['beta']:+.3f},"
            f" a+b {f['sum']:.3f}, a-b {f['diff']:+.3f}) SE_half {f['se_half']:.3f} SE_boot {f['se_boot']:.3f} "
            f"boot16-84 {np.round(f['boot_p16_p84'], 3).tolist()} | global p {g['p']:.3f} (a+b {g['sum']:.3f}) "
            f"| warning={bool(warn)}")
    ok = (abs(rows[0]["p"] - 0.5) < 0.05 and abs(rows[1]["p"] - 0.5) < 0.05
          and rows[3]["sum"] < 0.15 and rows[3]["warned"] and rows[4]["warned"])
    log(f"  PASS(p=0.5 when a=b>0; a=b=0 flagged via alpha+beta~0)={ok}  ({time.time() - t0:.1f}s)")
    RESULTS["test5"] = dict(pass_=ok, rows=rows)
    return ok


# --------------------------------------------------------------------------------------
# End-to-end CLI run on MRC files
# --------------------------------------------------------------------------------------
def test_cli():
    log("=" * 100)
    log("TEST E  end-to-end CLI on MRC files (cryoDRGN-style headers), (a,b,s) = (0.3,0.6,0.1) SYM mixed")
    t0 = time.time()
    maps, _ = build(0.3, 0.6, 0.1, seed=900)
    root = os.path.join(OUT, "synthetic_mrc")
    names = {"V": "concordant_eq32638", "M7": "c2swap_J487", "M6": "c2swap_J486"}
    for k, d in names.items():
        os.makedirs(os.path.join(root, d), exist_ok=True)
        for h, fn in (("full", "backproject.mrc"), ("a", "half_map_a.mrc"), ("b", "half_map_b.mrc")):
            rf.write_mrc(os.path.join(root, d, fn), maps[k][h], APIX)
    # header round-trip check
    arr, info = rf.read_mrc(os.path.join(root, names["V"], "backproject.mrc"))
    hdr_ok = info["axes"] == (1, 2, 3) and abs(info["apix"] - APIX) < 1e-4 and np.allclose(arr, maps["V"]["full"])
    out = os.path.join(OUT, "cli_run")
    cmd = [sys.executable, os.path.join(HERE, "register_fraction.py"),
           "--vdir", os.path.join(root, names["V"]), "--m7dir", os.path.join(root, names["M7"]),
           "--m6dir", os.path.join(root, names["M6"]), "--axis", "0.720", "-0.692", "0.061",
           "--apix", str(APIX), "--lowpass", "8", "--out", out]
    t1 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t1
    if pr.returncode != 0:
        log(pr.stdout[-3000:])
        log(pr.stderr[-3000:])
        RESULTS["cli"] = dict(pass_=False)
        return False
    R = json.load(open(os.path.join(out, "results.json")))
    p7 = R["fits"]["focused"]["M7"]["p"]
    ok = hdr_ok and abs(p7 - 1 / 3) <= 0.05 and all(
        os.path.exists(os.path.join(out, f)) for f in ("fig_slices.png", "fig_fit.png", "fig_axis.png", "report.txt"))
    log(f"  header ok {hdr_ok}; CLI wall time {dt:.1f}s; p7 focused {p7:.3f} (true 0.333); mapping "
        f"{R['convention']['best']['label']}; refined axis {R['global_search']['angle_refined_vs_nominal']:.2f} deg "
        f"from mapped u")
    log(f"  outputs in {out}")
    log(f"  PASS={ok}  ({time.time() - t0:.1f}s)")
    RESULTS["cli"] = dict(pass_=ok, wall_s=dt, p7=p7, out=out)
    return ok


if __name__ == "__main__":
    T = time.time()
    which = os.environ.get("TESTS", "0,1,2,3,4,5,E").split(",")
    fns = {"0": test0, "1": test1, "2": test2, "3": test3, "4": test4, "5": test5, "E": test_cli}
    status = {}
    for k in which:
        status[k] = fns[k]()
    log("=" * 100)
    log("SUMMARY " + "  ".join(f"test{k}={'PASS' if v else 'FAIL'}" for k, v in status.items())
        + f"   total {time.time() - T:.1f}s")
    with open(os.path.join(OUT, "test_results.json"), "w") as f:
        json.dump(rf.strip_private(RESULTS), f, indent=1, default=float)
