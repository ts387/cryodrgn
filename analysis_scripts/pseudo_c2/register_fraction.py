#!/usr/bin/env python3
"""
register_fraction.py -- what share of a pseudo-C2 "swapped" particle subset sits in the
concordant register?

Background and model
--------------------
Two refinements of a pseudo-C2 complex (MINA53 dimer + one MPP6) disagree for a subset of
particles by a ~180 deg rotation about a single map-frame axis u.  That subset is
backprojected (cryoDRGN backproject_voxel) once with each job's poses; an equally sized set
of particles on which both jobs agree defines the reference register:

    V  = concordant particles               (reference register)
    M7 = swapped particles, J487 poses
    M6 = the same swapped particles, J486 poses

Let C be the 180 deg rotation about u (through the box centre, optionally refined to the
pseudo-2-fold of V).  If a fraction a of the swapped particles is correctly posed by J487, a
fraction b by J486 and the rest carry no register information (particles that are themselves
symmetric, junk), then

    M7 ~ a V + b C(V) + (C-symmetric stuff) + noise
    M6 ~ b V + a C(V) + (C-symmetric stuff) + noise

and the quantity of interest is p = a / (a + b).  It is estimated by weighted least squares
inside a mask,

    M ~ alpha V + beta C(V) + gamma ,      p = alpha / (alpha + beta).

Split V and M into C-symmetric and C-antisymmetric parts: the antisymmetric part fixes
alpha - beta, the symmetric part fixes alpha + beta.  C-symmetric contaminants in M (symmetric
particles, junk) inflate alpha + beta and pull p towards 0.5.  This is why the script also
fits inside a FOCUSED mask, the region where V and C(V) differ (MPP6 and its symmetry mate).
There the symmetric contaminants carry little density, and the focused fit is the recommended
headline number (see the synthetic tests in test_register_fraction.py).

Review changes (2026-10-19; the unreviewed version is register_fraction.orig.py)
--------------------------------------------------------------------------------
* Third, C-symmetric regressor min(V, C(V)) (the dimer density common to both registers):
      M ~ alpha V + beta C(V) + delta min(V, C(V)) + gamma .
  Even the focused mask touches the protomer surface next to MPP6, so MPP6-less symmetric
  dimers leaked into alpha + beta.  In the reviewer's test, 92% such particles gave p = 0.52
  (true 0.625) with a tight SE and no warning; with the third regressor p = 0.63 and the
  weak-signal warning fires.  --no-common-regressor restores the 2-regressor fit, which is
  still reported for comparison.
* Scale-free amplitudes: kappa = scale of M relative to V on the C-symmetric dimer core;
  (alpha+beta)/kappa and (alpha-beta)/kappa replace the raw values in the weak-signal test and
  in the p bounds.  The raw ones assume V and M share an absolute scale; with M at 1.3x V, the
  old "bound" [0.888, 1] excluded the true p = 0.875.
* (alpha+beta)/kappa ~ 2 at p ~ 0.5 indicates MPP6 on both protomers (2:2); ~ 1 indicates a
  50:50 register mix.  This assumes V's MPP6 occupancy matches that of M's particles.
* Half-map noise ratio M7/V (same N, so a pure scale estimate).  It gives absolute shares of
  swapped particles that carry register-specific density, (alpha+beta)/ratio, and that have a
  dimer core, kappa/ratio.  Featureless junk lowers both but leaves (alpha+beta)/kappa near
  0.35; the weak-signal warning also tests the absolute share (< 0.25).
* Trade-off of the third regressor: if V itself contains a fraction eps of flipped particles,
  min(V, CV) carries eps (F + CF), and p is pushed away from 0.5 as
      p_fit = 0.5 + (p - 0.5) / (1 - 2 eps / r),   r = (alpha+beta)/kappa
  (the 2-regressor fit: r = 1).  Checked on test_register_fraction.py test 4: predicted
  1.064/0.714/0.286/1.251 against fitted 1.049/0.717/0.289/1.202.  The signature, p > 1 or
  beta < 0, is unchanged.  Test 4's 2-regressor prediction therefore no longer applies to
  its last case.
* fit_mixture / fit_antisym no longer divide by zero for a constant M or an exactly
  C-symmetric V; the sign test is referenced to the solvent level (offset-proof).

Conventions (verified, not assumed)
-----------------------------------
cryoDRGN 4.3.1 backproject_voxel accumulates Fourier samples at flat index
z*res^2 + y*res + x and then applies ihtn_center.  So the array written to backproject.mrc is
indexed [z, y, x] of the same frame in which the pose matrices act (rot maps map -> image,
ff_coord = lattice.coords @ rot), with the origin at index D/2.  For two poses R7, R6 of the
same image the map-frame operator relating the two reconstructions is R7^T R6, so u must be
the axis of R7^T R6 (equivalently R6^T R7), NOT of R6 R7^T.

The script does not trust that.  It
  (1) tests all 24 signed-permutation mappings of u into array-index order (each with centre
      D/2 and D/2 - 0.5) and keeps the one that maximises corr(V, C(V)) in the mask, and
  (2) runs an independent global search for the best 2-fold of V (about 2.5 deg grid over the
      hemisphere, then Nelder-Mead refinement of axis and axis offset), and reports the
      angle between that axis and the chosen mapping of u.
A mapping is only defined up to an overall sign (P and -P give the same axis line), so
"[+z,+y,+x]" and "[-z,-y,-x]" are the same class.  Caveat for this u: |u_x| ~ |u_y| and
u_z ~ 0, so the classes [+z,-x,-y] (2.3 deg away) and [-z,+y,+x] (7 deg away) are nearly the
same operator.  The search cannot tell them apart, and does not need to.

Uncertainty
-----------
* half-map spread: refit with M half A / half B (full V) and with V half A / half B (full M).
  SE ~ 0.5 * sqrt((pA - pB)^2 + (pVA - pVB)^2).  This is a single-degree-of-freedom
  estimate, so treat it as rough.
* noise-injection bootstrap: noise with the exact (anisotropic) amplitude spectrum of
  (half A - half B)/2 and random phases is added to V and M, and the fit is repeated.
  SE = std over draws, plus a 16-84 percentile interval.
Note: if all swapped particles differ by the same operator, M6 is just C(M7) and p6 = 1 - p7
follows automatically.  So M6 is a consistency check of the operator, not an independent
replicate.  The script reports corr(M6, C(M7)) so you can see this.

Usage
-----
    python register_fraction.py \
        --vdir  concordant_eq32638 --m7dir c2swap_J487 --m6dir c2swap_J486 \
        --axis 0.720 -0.692 0.061 --apix 2.609375 --lowpass 8 --out regfrac_out

Each directory must contain backproject.mrc, half_map_a.mrc and half_map_b.mrc
(cryoDRGN backproject_voxel output).  Outputs in --out: results.json, report.txt,
fig_slices.png, fig_fit.png, fig_axis.png.
Dependencies: numpy, scipy, matplotlib; mrcfile optional (a minimal reader is built in).
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, optimize
from scipy.spatial import cKDTree

AXNAMES = "xyz"
HYPOTHESIS_LABEL = "[+z,+y,+x]"  # array axes (0,1,2) <- map (z,y,x): cryoDRGN default
WORKERS = -1  # scipy.fft threads


# --------------------------------------------------------------------------------------
# MRC I/O
# --------------------------------------------------------------------------------------
def _read_mrc_minimal(path):
    """Tiny MRC2014 reader (modes 0,1,2,6,12) used when mrcfile is unavailable."""
    with open(path, "rb") as f:
        hdr = f.read(1024)
        stamp = hdr[212:214]
        endian = ">" if stamp[:1] == b"\x11" else "<"
        ints = np.frombuffer(hdr, dtype=endian + "i4", count=56)
        flts = np.frombuffer(hdr, dtype=endian + "f4", count=56)
        nx, ny, nz, mode = (int(v) for v in ints[:4])
        mx = int(ints[7])
        cella_x = float(flts[10])
        axes = tuple(int(v) for v in ints[16:19])
        nsymbt = int(ints[23])
        origin = tuple(float(v) for v in flts[49:52])
        nstart = tuple(int(v) for v in ints[4:7])
        dt = {0: "i1", 1: "i2", 2: "f4", 6: "u2", 12: "f2"}[mode]
        f.seek(1024 + nsymbt)
        data = np.frombuffer(f.read(), dtype=endian + dt, count=nx * ny * nz)
    data = data.reshape(nz, ny, nx).astype(np.float32)
    apix = cella_x / mx if mx > 0 and cella_x > 0 else None
    return data, apix, axes, origin, nstart


def read_mrc(path):
    """Return (array as stored on disk [sections, rows, cols], info dict)."""
    try:
        import mrcfile

        with mrcfile.open(path, permissive=True) as m:
            data = np.asarray(m.data, dtype=np.float32).copy()
            vs = float(m.voxel_size.x)
            h = m.header
            info = dict(
                apix=vs if vs > 0 else None,
                axes=(int(h.mapc), int(h.mapr), int(h.maps)),
                origin=(float(h.origin.x), float(h.origin.y), float(h.origin.z)),
                nstart=(int(h.nxstart), int(h.nystart), int(h.nzstart)),
            )
    except ImportError:
        data, apix, axes, origin, nstart = _read_mrc_minimal(path)
        info = dict(apix=apix, axes=axes, origin=origin, nstart=nstart)
    return data, info


def write_mrc(path, arr, apix):
    """Write a float32 volume stored [z, y, x] with mapc,mapr,maps = 1,2,3 (like cryoDRGN)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        import mrcfile

        with mrcfile.new(path, overwrite=True) as m:
            m.set_data(arr)
            m.voxel_size = apix
    except ImportError:  # minimal writer
        nz, ny, nx = arr.shape
        ints = np.zeros(56, "<i4")
        flts = ints.view("<f4")
        ints[:4] = (nx, ny, nz, 2)
        ints[7:10] = (nx, ny, nz)
        flts[10:13] = (apix * nx, apix * ny, apix * nz)
        flts[13:16] = 90.0
        ints[16:19] = (1, 2, 3)
        flts[19:22] = (arr.min(), arr.max(), arr.mean())
        hdr = bytearray(ints.tobytes()) + bytearray(1024 - 224)
        hdr[208:212] = b"MAP "
        hdr[212:214] = b"\x44\x44"
        flts2 = np.frombuffer(bytes(hdr[:224]), "<f4").copy()
        flts2[54] = arr.std()
        hdr[:224] = flts2.tobytes()
        with open(path, "wb") as f:
            f.write(bytes(hdr))
            f.write(arr.tobytes())


def load_dir(d, need_halves=True):
    """Load backproject.mrc (+ half maps) from a cryoDRGN backproject_voxel folder."""
    out, infos = {}, {}
    for key, fn in (("full", "backproject.mrc"), ("a", "half_map_a.mrc"), ("b", "half_map_b.mrc")):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out[key], infos[key] = read_mrc(p)
        elif key == "full" or need_halves:
            raise FileNotFoundError(p)
        else:
            out[key] = None
    return out, infos["full"]


# --------------------------------------------------------------------------------------
# Fourier helpers: low-pass filter and cubic B-spline coefficients from the same FFT
# --------------------------------------------------------------------------------------
def freq_radius(shape, apix):
    """|s| in 1/A on the rfftn grid of a real volume of this shape."""
    kz = sfft.fftfreq(shape[0], d=apix)
    ky = sfft.fftfreq(shape[1], d=apix)
    kx = sfft.rfftfreq(shape[2], d=apix)
    return np.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2)


def lowpass_filter(shape, apix, res_A, width_px=3.0):
    """Raised-cosine low-pass: 1 below res, 0 above; edge width_px Fourier pixels centred on 1/res."""
    s = freq_radius(shape, apix)
    sc = 1.0 / res_A
    ds = width_px / (shape[0] * apix)
    f = np.clip((sc + ds / 2 - s) / ds, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * f)).astype(np.float32)


def bspline3_transfer(shape):
    """DTFT of the sampled cubic B-spline (2 + cos w)/3 per axis, on the rfftn grid.
    Dividing a spectrum by it gives (periodic) cubic-spline coefficients, so maps can be
    interpolated with map_coordinates(order=3, prefilter=False) with no extra prefilter pass."""
    wz = 2 * np.pi * sfft.fftfreq(shape[0])
    wy = 2 * np.pi * sfft.fftfreq(shape[1])
    wx = 2 * np.pi * sfft.rfftfreq(shape[2])
    b = lambda w: (2.0 + np.cos(w)) / 3.0  # noqa: E731
    return (b(wz)[:, None, None] * b(wy)[None, :, None] * b(wx)[None, None, :]).astype(np.float32)


class Spectra:
    """Caches the rfftn of each raw map so that low-passing at several resolutions is cheap."""

    def __init__(self, shape, apix):
        self.shape, self.apix = tuple(shape), apix
        self.bden = bspline3_transfer(shape)
        self._filters = {}
        self._ffts = {}

    def filt(self, res_A):
        if res_A not in self._filters:
            self._filters[res_A] = lowpass_filter(self.shape, self.apix, res_A)
        return self._filters[res_A]

    def fft(self, key, vol):
        if key not in self._ffts:
            self._ffts[key] = sfft.rfftn(vol.astype(np.float32), workers=WORKERS).astype(np.complex64)
        return self._ffts[key]

    def lowpass(self, key, vol, res_A, coeffs=False):
        h = self.fft(key, vol) * self.filt(res_A)
        lp = sfft.irfftn(h, s=self.shape, workers=WORKERS).astype(np.float32)
        if not coeffs:
            return lp
        c = sfft.irfftn(h / self.bden, s=self.shape, workers=WORKERS).astype(np.float32)
        return lp, c


# --------------------------------------------------------------------------------------
# The 2-fold operator
# --------------------------------------------------------------------------------------
class TwoFold:
    """C(x) = R (x - c) + c + T  in ARRAY-INDEX coordinates (x = (i0, i1, i2)).

    R = 2 u u^T - I is the 180 deg rotation about the unit axis u (array order), c is the
    rotation centre and T a translation kept perpendicular to u, so that C is an involution
    (C o C = identity).  T corresponds to the 2-fold axis passing through c + T/2.
    Because C is its own inverse, (C V)(x) = V(C(x))."""

    def __init__(self, axis, centre, T=(0.0, 0.0, 0.0)):
        u = np.asarray(axis, float)
        self.u = u / np.linalg.norm(u)
        self.c = np.broadcast_to(np.asarray(centre, float), (3,)).copy()
        T = np.asarray(T, float)
        self.T = T - self.u * np.dot(T, self.u)
        self.R = 2.0 * np.outer(self.u, self.u) - np.eye(3)

    def points(self, X):
        return (X - self.c) @ self.R.T + self.c + self.T

    def sample(self, coef, X, order=3):
        """Values of C(vol) at points X.  coef: cubic coefficients (order 3) or the map (order 1)."""
        Y = self.points(X)
        return ndimage.map_coordinates(coef, Y.T, order=order, prefilter=False, mode="grid-wrap")

    def volume(self, coef, order=3, mode="grid-wrap"):
        """Full-grid C(vol).  order=3 expects spline coefficients; order=1 expects the map."""
        offset = self.c + self.T - self.R @ self.c
        return ndimage.affine_transform(coef, self.R, offset=offset, order=order, prefilter=False,
                                        mode=mode).astype(np.float32)

    def describe(self, apix):
        return dict(axis_array=[round(float(v), 5) for v in self.u],
                    axis_map_xyz_if_zyx=[round(float(v), 5) for v in self.u[::-1]],
                    centre=[float(v) for v in self.c],
                    T_px=[round(float(v), 3) for v in self.T],
                    axis_offset_A=round(float(np.linalg.norm(self.T) / 2 * apix), 2))


def line_angle(a, b):
    """Angle (deg) between two axis LINES (sign-insensitive)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


# --------------------------------------------------------------------------------------
# Axis-mapping candidates
# --------------------------------------------------------------------------------------
def mapping_classes():
    """The 48 signed permutations P (array_vec = P @ map_vec) grouped into 24 classes {P, -P}.
    Returns list of (canonical_label, alt_label, P) with the first row's sign made positive."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            if signs[0] < 0:
                continue
            P = np.zeros((3, 3))
            for i in range(3):
                P[i, perm[i]] = signs[i]
            lab = "[" + ",".join(("+" if signs[i] > 0 else "-") + AXNAMES[perm[i]] for i in range(3)) + "]"
            alt = "[" + ",".join(("-" if signs[i] > 0 else "+") + AXNAMES[perm[i]] for i in range(3)) + "]"
            out.append((lab, alt, P))
    return out


def parse_mapping(label):
    """'[+z,+y,+x]' -> P (array_vec = P @ map_vec)."""
    toks = label.strip("[] ").split(",")
    P = np.zeros((3, 3))
    for i, t in enumerate(toks):
        P[i, AXNAMES.index(t[1])] = 1.0 if t[0] == "+" else -1.0
    return P


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
def soft_edge(binary, apix, soft_A):
    if soft_A <= 0:
        return binary.astype(np.float32)
    d = ndimage.distance_transform_edt(~binary) * apix
    m = np.where(binary, 1.0, np.where(d < soft_A, 0.5 * (1 + np.cos(np.pi * d / soft_A)), 0.0))
    return m.astype(np.float32)


def dilate(binary, apix, dil_A):
    if dil_A <= 0:
        return binary
    return ndimage.distance_transform_edt(~binary) * apix <= dil_A


def largest_component(binary):
    lab, n = ndimage.label(binary)
    if n <= 1:
        return binary
    sizes = ndimage.sum(binary, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def envelope_mask(vol_lp, apix, kda=125.0, threshold=None, dilate_A=6.0, soft_A=6.0):
    """Threshold (absolute, or chosen so the binary volume equals kda * 1.21 A^3/Da),
    keep the largest connected component, dilate, cosine soft edge."""
    if threshold is None:
        nvox = int(round(kda * 1000.0 * 1.21 / apix ** 3))
        nvox = max(10, min(nvox, vol_lp.size - 1))
        threshold = float(np.partition(vol_lp.ravel(), -nvox)[-nvox])
    binary = largest_component(vol_lp >= threshold)
    binary = dilate(binary, apix, dilate_A)
    return soft_edge(binary, apix, soft_A), float(threshold)


def focus_mask(V_lp, CV_lp, env, apix, level=0.3, excl=0.5, dilate_A=3.0, soft_A=4.0, smooth_px=1.0,
               background=0.0):
    """Register-specific ("exclusive") density: voxels where
         |V - CV| > level * (99.5th pct of |V - CV| inside env)            (big difference), and
         |V - CV| > excl  * max(V - bg, CV - bg, 0)                         (density present in ONE
                                                                             register only).
    The second condition keeps MPP6 and its symmetry mate and drops regions where both
    protomers have density that differs only partly (protomer non-identity).  Particles that
    are themselves symmetric put density there, which would leak into alpha + beta and bias p
    toward 0.5.  Use excl=0 for a plain |V - CV| threshold.
    |V - CV| is invariant under C, so the mask is (up to interpolation) C-symmetric."""
    D = ndimage.gaussian_filter(np.abs(V_lp - CV_lp), smooth_px)
    S = ndimage.gaussian_filter(np.maximum(np.maximum(V_lp, CV_lp) - background, 0), smooth_px)
    inside = env > 0.5
    ref = float(np.percentile(D[inside], 99.5))
    binary = (D > level * ref) & (D > excl * S) & inside
    lab, n = ndimage.label(binary)  # drop specks
    if n > 1:
        sizes = ndimage.sum(binary, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= 20)[0])
        binary = keep if keep.any() else binary
    binary = dilate(binary, apix, dilate_A)
    return np.minimum(soft_edge(binary, apix, soft_A), env).astype(np.float32)


def symmetrize_mask(m, op):
    return np.maximum(m, np.clip(op.volume(m, order=1, mode="constant"), 0, 1)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Correlation / fitting
# --------------------------------------------------------------------------------------
def wcorr(a, b, w):
    sw = w.sum()
    da = a - (w * a).sum() / sw
    db = b - (w * b).sum() / sw
    den = math.sqrt(float((w * da * da).sum() * (w * db * db).sum()))
    return float((w * da * db).sum() / den) if den > 0 else float("nan")


def fit_mixture(m, v, cv, w, vmin=None):
    """Weighted LS  m ~ alpha v + beta cv [+ delta min(v, cv)] + gamma;  p = alpha/(alpha+beta).

    REVIEW FIX: the optional third regressor vmin = min(V, CV) is the density common to both
    registers (the dimer without MPP6; it is C-symmetric).  With it, C-symmetric particles
    without MPP6 load on delta instead of inflating alpha + beta.  Without it, they pull p
    toward 0.5 even in the focused mask: in the reviewer's test, 92% MPP6-less dimers gave
    p = 0.52 with alpha + beta = 0.29, against a true p of 0.625 and k(a+b) = 0.056.  Equivalently
    m ~ (alpha+beta+delta) Vs + (alpha-beta) Va - delta |Va|, with Vs, Va the symmetric and
    antisymmetric halves of V."""
    sw = np.sqrt(w)
    cols = [v, cv] + ([vmin] if vmin is not None else []) + [np.ones_like(v)]
    X = np.stack(cols, 1) * sw[:, None]
    y = m * sw
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = [float(c) for c in coef]
    a, b, g = coef[0], coef[1], coef[-1]
    dlt = coef[2] if vmin is not None else 0.0
    res = y - X @ coef
    mbar = (w * m).sum() / w.sum()
    tss = float((w * (m - mbar) ** 2).sum())  # REVIEW FIX: constant M in the mask used to raise ZeroDivisionError
    r2 = 1.0 - float((res ** 2).sum()) / tss if tss > 0 else float("nan")
    s = a + b
    return dict(alpha=a, beta=b, gamma=g, delta=dlt, p=(a / s if s != 0 else float("nan")),
                sum=s, diff=a - b, r2=r2)


def fit_antisym(m, cm, v, cv, w, vmin=None):
    """Antisymmetric-only estimate: (m - Cm) = d (v - Cv)  (weighted, no intercept), and the
    symmetric amplitude (m + Cm) = s (v + Cv) [+ 2 delta min(v, cv)] + g.  p = 1/2 + d / (2 s)."""
    xa, ya = v - cv, m - cm
    den = float((w * xa * xa).sum())  # REVIEW FIX: V exactly C-symmetric in the mask -> no division by zero
    d = float((w * xa * ya).sum()) / den if den > 0 else float("nan")
    xs, ys = v + cv, m + cm
    sw = np.sqrt(w)
    cols = [xs] + ([vmin] if vmin is not None else []) + [np.ones_like(xs)]
    A = np.stack(cols, 1) * sw[:, None]
    sol, *_ = np.linalg.lstsq(A, ys * sw, rcond=None)
    s = float(sol[0])
    return dict(diff=d, sum=s, p=0.5 + d / (2 * s) if s != 0 else float("nan"))


# --------------------------------------------------------------------------------------
# Axis searches
# --------------------------------------------------------------------------------------
def selfcorr(src, X, v0, w, op, order):
    return wcorr(v0, op.sample(src, X, order=order), w)


def best_offset_fft(Vlp, m, axis, centre, max_shift_px=12.0):
    """For the 2-fold about `axis` through `centre`, find the translation T (C(x) = R(x-c)+c+T)
    that best superimposes V on itself, by one FFT cross-correlation.
    With W(y) = V(R(y-c)+c):  V(C x) = W(x + R T), so T = R s*, where s* maximises
    sum_x m(x) V(x) W(x+s).  The component of T along the axis (a screw) is discarded."""
    op0 = TwoFold(axis, centre)
    W = op0.volume(Vlp, order=1)
    mu = float((m * Vlp).sum() / m.sum())
    A = m * (Vlp - mu)
    B = W - mu
    xc = sfft.irfftn(np.conj(sfft.rfftn(A, workers=WORKERS)) * sfft.rfftn(B, workers=WORKERS), s=Vlp.shape,
                     workers=WORKERS)
    n = Vlp.shape[0]
    f = np.fft.fftfreq(n) * n
    rr = f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
    xc = np.where(rr <= max_shift_px ** 2, xc, -np.inf)
    k = np.array(np.unravel_index(int(np.argmax(xc)), xc.shape))
    s = np.where(k > n // 2, k - n, k).astype(float)
    for ax in range(3):  # parabolic sub-voxel refinement
        e = np.zeros(3, int)
        e[ax] = 1
        fm, f0, fp = (float(xc[tuple((k + d * e) % n)]) for d in (-1, 0, 1))
        den = fm - 2 * f0 + fp
        if np.isfinite(den) and den < 0:
            s[ax] += 0.5 * (fm - fp) / den
    T = op0.R @ s
    return TwoFold(axis, centre, T).T


def convention_search(Vcoef, X, v0, w, u_map, centres, Vlp=None, m=None, with_offset=True):
    """Score every axis-mapping class by corr(V, C(V)) inside the mask:
       corr_centre : 2-fold through each candidate centre (D/2, D/2-0.5), no offset;
       corr_offset : 2-fold through D/2 plus the best perpendicular offset (FFT search), so an
                     off-centre pseudo-2-fold does not make the right mapping lose.
    Rows are ranked by `corr` = corr_offset if with_offset else the best corr_centre."""
    rows = []
    for lab, alt, P in mapping_classes():
        ua = P @ u_map
        cc = {float(c): selfcorr(Vcoef, X, v0, w, TwoFold(ua, c), 3) for c in centres}
        cbest = max(cc, key=cc.get)
        row = dict(label=lab, alt=alt, axis_array=ua.tolist(), corr_centre=cc, centre=cbest,
                   corr=cc[cbest], T=[0.0, 0.0, 0.0])
        if with_offset and Vlp is not None:
            T = best_offset_fft(Vlp, m, ua, centres[0])
            row["corr_offset"] = selfcorr(Vcoef, X, v0, w, TwoFold(ua, centres[0], T), 3)
            row["T"] = T.tolist()
            if row["corr_offset"] > row["corr"]:
                row.update(corr=row["corr_offset"], centre=float(centres[0]))
            else:
                row["T"] = [0.0, 0.0, 0.0]
        rows.append(row)
    rows.sort(key=lambda r: -r["corr"])
    return rows


def self_rotation_search(sp, Vraw, env, step_deg=2.5, band=(15.0, 6.0), rmin_A=5.0, rmax_A=35.0,
                         npts=10000, rng=None):
    """Offset-INVARIANT coarse search (crystallographic self-rotation function): the Patterson
    map of the band-passed, masked V is 2-fold symmetric about the origin for a 2-fold of V
    through ANY point.  Short vectors (rmin..rmax A) of the 15-6 A band carry internal
    structure rather than the envelope shape; low resolution alone just finds principal axes."""
    shape, apix = sp.shape, sp.apix
    hi = max(band[1], 2.2 * apix)
    h = sp.fft("V", Vraw) * sp.filt(hi) * (1 - sp.filt(band[0]))
    A = env * sfft.irfftn(h, s=shape, workers=WORKERS)
    P = np.fft.fftshift(sfft.irfftn(np.abs(sfft.rfftn(A, workers=WORKERS)) ** 2, s=shape, workers=WORKERS))
    P = P.astype(np.float32)
    c = shape[0] // 2
    r = np.sqrt(((np.indices(shape) - c) ** 2).sum(0)) * apix
    X = np.argwhere((r > rmin_A) & (r < rmax_A)).astype(float)
    rng = rng or np.random.default_rng(0)
    if len(X) > npts:
        X = X[rng.choice(len(X), npts, replace=False)]
    return coarse_axis_search(P, X, np.ones(len(X)), float(c), step_deg)


def fibonacci_hemisphere(step_deg):
    n = int(math.ceil(4 * math.pi / math.radians(step_deg) ** 2))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(phi), np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th)], 1)
    return d[d[:, 0] >= 0]


def coarse_axis_search(Vlp, X, w, centre, step_deg=2.5, chunk=48):
    """Correlation of V with its 180-deg rotation about every grid direction (order-1 interp)."""
    dirs = fibonacci_hemisphere(step_deg)
    v0 = Vlp[tuple(np.round(X).astype(int).T)]
    c = np.asarray(centre, float)
    Xc = X - c
    corr = np.empty(len(dirs))
    for s in range(0, len(dirs), chunk):
        dd = dirs[s:s + chunk]
        # R x = 2 u (u.x) - x  for each direction u
        proj = Xc @ dd.T  # (N, k)
        Y = 2 * proj.T[:, :, None] * dd[:, None, :] - Xc[None] + c  # (k, N, 3)
        vals = ndimage.map_coordinates(Vlp, Y.reshape(-1, 3).T, order=1, mode="grid-wrap")
        vals = vals.reshape(len(dd), -1)
        for k in range(len(dd)):
            corr[s + k] = wcorr(v0, vals[k], w)
    # local maxima on the sphere of lines
    both = np.concatenate([dirs, -dirs])
    cc = np.concatenate([corr, corr])
    tree = cKDTree(both)
    r = 2 * math.sin(math.radians(1.8 * step_deg) / 2)
    peaks = []
    for i in range(len(dirs)):
        nb = tree.query_ball_point(dirs[i], r)
        if corr[i] >= cc[nb].max() - 1e-12:
            peaks.append(i)
    peaks.sort(key=lambda i: -corr[i])
    return dirs, corr, peaks


def tangent_basis(u):
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def refine_twofold(Vcoef, X, v0, w, centre, axis0, fit_axis=True, fit_offset=True, T0=None):
    """Nelder-Mead on axis direction (tangent plane) and/or offset T (perpendicular to axis)."""
    axis0 = np.asarray(axis0, float) / np.linalg.norm(axis0)
    e1, e2 = tangent_basis(axis0)
    T0 = np.zeros(3) if T0 is None else np.asarray(T0, float)

    def make(prm):
        ax, i = axis0, 0
        if fit_axis:
            ax = axis0 + prm[0] * e1 + prm[1] * e2
            i = 2
        T = T0.copy()
        if fit_offset:
            f1, f2 = tangent_basis(ax)
            T = T0 + prm[i] * f1 + prm[i + 1] * f2
        return TwoFold(ax, centre, T)

    n = 2 * int(fit_axis) + 2 * int(fit_offset)
    if n == 0:
        op = make([])
        return op, selfcorr(Vcoef, X, v0, w, op, 3)
    steps = ([0.02, 0.02] if fit_axis else []) + ([0.5, 0.5] if fit_offset else [])
    simplex = np.vstack([np.zeros(n)] + [np.eye(n)[k] * steps[k] for k in range(n)])
    res = optimize.minimize(lambda prm: -selfcorr(Vcoef, X, v0, w, make(prm), 3), np.zeros(n),
                            method="Nelder-Mead",
                            options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-6, maxiter=600))
    op = make(res.x)
    return op, -float(res.fun)


# --------------------------------------------------------------------------------------
# Noise model for the bootstrap
# --------------------------------------------------------------------------------------
class NoiseModel:
    """Gaussian noise whose expected |FFT|^2 equals that of (half_a - half_b)/2 (the noise of
    the full map), low-passed like the data.  Phases are random, the amplitude spectrum keeps
    its anisotropy."""

    def __init__(self, spectra, a, b, res_A):
        h = sfft.rfftn(((a - b) / 2).astype(np.float32), workers=WORKERS)
        self.amp = (np.abs(h) * spectra.filt(res_A)).astype(np.float32)
        self.spectra = spectra

    def draw(self, rng, coeffs=False):
        # white real noise -> rfftn has E|h|^2 = N_vox and exact Hermitian symmetry
        W = rng.standard_normal(self.spectra.shape, dtype=np.float32)
        h = sfft.rfftn(W, workers=WORKERS) * (self.amp / np.sqrt(W.size))
        n = sfft.irfftn(h, s=self.spectra.shape, workers=WORKERS).astype(np.float32)
        if not coeffs:
            return n
        c = sfft.irfftn(h / self.spectra.bden, s=self.spectra.shape, workers=WORKERS).astype(np.float32)
        return n, c


# --------------------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------------------
DEFAULTS = dict(lowpass=8.0, mask_lowpass=15.0, mask_kda=125.0, mask_threshold=None, mask_dilate=6.0,
                mask_soft=6.0, focus_level=0.3, focus_excl=0.5, focus_dilate=3.0, focus_soft=4.0,
                centre="auto", mapping="auto", conv_offset=True, operator="refined", global_search=True,
                coarse_lowpass=12.0, grid_deg=2.5, srf_band=(15.0, 6.0), srf_rmax=35.0, n_boot=50, seed=0,
                sweep_lowpass=(5.0, 8.0, 12.0),
                sweep_focus=((0.3, 0.0), (0.5, 0.0), (0.2, 0.5), (0.3, 0.5), (0.3, 0.7)),
                diagnostics=True, max_axis_dev=6.0, common_regressor=True)


def _fit_set(Mlp, idx, v, cv, w, vmin=None):
    return fit_mixture(Mlp.ravel()[idx], v, cv, w, vmin)


def _vmin(v, cv, use):
    return np.minimum(v, cv) if use else None


def evaluate_masks(L, op, masks, opts, rng, n_boot, spectra=None, noise=None, log=print):
    """Fit M7 and M6 inside each mask; half-map spread; bootstrap.  L holds low-passed maps."""
    out = {}
    use3 = bool((opts or {}).get("common_regressor", True))  # REVIEW FIX: third regressor min(V, CV)
    CV = op.volume(L["V_coef"])
    for mname, m in masks.items():
        idx = np.flatnonzero(m > 1e-3)
        X = np.stack(np.unravel_index(idx, m.shape), 1).astype(float)
        w = m.ravel()[idx].astype(np.float64)
        v = L["V"].ravel()[idx].astype(np.float64)
        cv = CV.ravel()[idx].astype(np.float64)
        vmin = _vmin(v, cv, use3)
        res_m = dict(n_vox_support=int(idx.size), sum_w=float(w.sum()), corr_V_CV=wcorr(v, cv, w),
                     asym_frac=float(np.sqrt((w * (v - cv) ** 2).sum() / (w * (v + cv) ** 2).sum())))
        have_vh = L.get("Va") is not None
        if have_vh:
            va, vb = L["Va"].ravel()[idx], L["Vb"].ravel()[idx]
            cva, cvb = op.sample(L["Va_coef"], X), op.sample(L["Vb_coef"], X)
        for M in ("M7", "M6"):
            if M not in L:
                continue
            r = _fit_set(L[M], idx, v, cv, w, vmin)
            if use3:  # the original 2-regressor fit, for comparison
                r2_ = _fit_set(L[M], idx, v, cv, w)
                r["two_reg"] = dict(p=r2_["p"], sum=r2_["sum"], alpha=r2_["alpha"], beta=r2_["beta"])
            cm = op.sample(L[M + "_coef"], X)
            r["antisym"] = fit_antisym(L[M].ravel()[idx], cm, v, cv, w, vmin)
            # half-map spread
            if L.get(M + "a") is not None:
                pa = _fit_set(L[M + "a"], idx, v, cv, w, vmin)["p"]
                pb = _fit_set(L[M + "b"], idx, v, cv, w, vmin)["p"]
                r["p_halfM"] = [pa, pb]
                se2 = (pa - pb) ** 2
                if have_vh:
                    mv = L[M].ravel()[idx]
                    pva = fit_mixture(mv, va, cva, w, _vmin(va, cva, use3))["p"]
                    pvb = fit_mixture(mv, vb, cvb, w, _vmin(vb, cvb, use3))["p"]
                    r["p_halfV"] = [pva, pvb]
                    se2 += (pva - pvb) ** 2
                r["se_half"] = 0.5 * math.sqrt(se2)
            res_m[M] = r
        out[mname] = dict(res_m, _idx=idx, _X=X, _w=w, _v=v, _cv=cv)

    # bootstrap (noise draws shared across masks)
    if n_boot and noise:
        boots = {mn: {M: [] for M in ("M7", "M6") if M in L} for mn in masks}
        for k in range(n_boot):
            nV, nVc = noise["V"].draw(rng, coeffs=True)
            nM = {M: noise[M].draw(rng) for M in ("M7", "M6") if M in noise}
            for mn, d in out.items():
                idx, X, w = d["_idx"], d["_X"], d["_w"]
                v = d["_v"] + nV.ravel()[idx]
                cv = d["_cv"] + op.sample(nVc, X)
                vmin = _vmin(v, cv, use3)
                for M in nM:
                    mv = L[M].ravel()[idx] + nM[M].ravel()[idx]
                    boots[mn][M].append(fit_mixture(mv, v, cv, w, vmin)["p"])
        for mn in boots:
            for M, arr in boots[mn].items():
                arr = np.asarray(arr)
                out[mn][M]["se_boot"] = float(np.std(arr, ddof=1))
                out[mn][M]["boot_p16_p84"] = [float(np.percentile(arr, 16)), float(np.percentile(arr, 84))]
                out[mn][M]["boot_frac_outside_0_1"] = float(np.mean((arr < 0) | (arr > 1)))
    return out, CV


def core_scale(L, op, env, foc, M):
    """REVIEW ADDITION.  Scale of M relative to V on the C-symmetric dimer core: weighted LS
    (M + CM)/2 ~ kappa (V + CV)/2 + g with weights env * (1 - focused mask).  Under the model,
    kappa = k (a + b + s_dimer), where k is the absolute scale of M relative to V and s_dimer
    the share of symmetric dimer-like particles.  (alpha + beta)/kappa and (alpha - beta)/kappa
    are therefore free of the V/M scale, unlike the raw alpha + beta."""
    wc = env * (1.0 - np.clip(foc, 0.0, 1.0))
    idx = np.flatnonzero(wc > 1e-3)
    X = np.stack(np.unravel_index(idx, wc.shape), 1).astype(float)
    w = wc.ravel()[idx].astype(np.float64)
    xs = 0.5 * (L["V"].ravel()[idx] + op.sample(L["V_coef"], X))
    ys = 0.5 * (L[M].ravel()[idx] + op.sample(L[M + "_coef"], X))
    sw = np.sqrt(w)
    A = np.stack([xs, np.ones_like(xs)], 1) * sw[:, None]
    (k, _), *_ = np.linalg.lstsq(A, ys * sw, rcond=None)
    return float(k)


def strip_private(d):
    if isinstance(d, dict):
        return {str(k): strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (list, tuple)):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def run_analysis(maps, apix, u_map, opts=None, log=print):
    """maps: {'V': {'full','a','b'}, 'M7': {...}, 'M6': {...}} arrays indexed as stored.
    Returns (results dict, arrays dict for plotting)."""
    o = dict(DEFAULTS)
    o.update(opts or {})
    t0 = time.time()
    rng = np.random.default_rng(o["seed"])
    u_map = np.asarray(u_map, float)
    u_map = u_map / np.linalg.norm(u_map)
    shape = maps["V"]["full"].shape
    if len(set(shape)) != 1:
        raise ValueError(f"non-cubic map {shape}")
    for k in maps:
        for h in ("full", "a", "b"):
            if maps[k].get(h) is not None and maps[k][h].shape != shape:
                raise ValueError(f"shape mismatch {k}/{h}: {maps[k][h].shape} vs {shape}")
    D = shape[0]
    R = dict(inputs=dict(shape=list(shape), apix=apix, u_map_xyz=u_map.tolist(),
                         options={k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}))
    sp = Spectra(shape, apix)

    # ---- sign check (cryoDRGN inverts data by default -> protein positive)
    Vm = sp.lowpass("V", maps["V"]["full"], o["mask_lowpass"])
    rr = np.sqrt(((np.indices(shape) - D / 2) ** 2).sum(0))
    # REVIEW FIX: reference to the solvent level, so a constant offset in V cannot flip the sign test
    # (raw min/max compared before: an offset below about -0.56 x peak inverted every map).
    inner = Vm[rr < 0.4 * D] - float(np.median(Vm[rr >= 0.4 * D]))
    sign = 1.0
    if -inner.min() > 1.5 * inner.max():
        sign = -1.0
        log("WARNING: V looks inverted (protein negative); multiplying ALL maps by -1 (p is unaffected).")
    R["sign"] = sign
    if sign < 0:
        maps = {k: {h: (None if a is None else -a) for h, a in d.items()} for k, d in maps.items()}
        sp = Spectra(shape, apix)
        Vm = -Vm

    # ---- low-pass everything identically
    def lowpass_all(res):
        L = {}
        L["V"], L["V_coef"] = sp.lowpass("V", maps["V"]["full"], res, coeffs=True)
        for h in ("a", "b"):
            a = maps["V"].get(h)
            if a is not None:
                L["V" + h], L["V" + h + "_coef"] = sp.lowpass("V" + h, a, res, coeffs=True)
        for M in ("M7", "M6"):
            if M not in maps:
                continue
            L[M], L[M + "_coef"] = sp.lowpass(M, maps[M]["full"], res, coeffs=True)
            for h in ("a", "b"):
                a = maps[M].get(h)
                L[M + h] = None if a is None else sp.lowpass(M + h, a, res)
        return L

    L = lowpass_all(o["lowpass"])

    # ---- envelope mask from V
    env, thr = envelope_mask(Vm, apix, kda=o["mask_kda"], threshold=o["mask_threshold"],
                             dilate_A=o["mask_dilate"], soft_A=o["mask_soft"])
    bg = float(np.median(Vm[env == 0])) if (env == 0).any() else 0.0
    R["mask"] = dict(threshold=thr, background=bg, n_vox_env_gt_half=int((env > 0.5).sum()))
    log(f"[mask] envelope threshold {thr:.4g} (mask_lowpass {o['mask_lowpass']} A, background {bg:.3g}), "
        f"{(env > 0.5).sum()} voxels > 0.5")

    # points for the convention search / refinement: envelope voxels > 0.5 (+ a subsample for Nelder-Mead)
    Xe = np.argwhere(env > 0.5).astype(float)
    we = np.ones(len(Xe))
    v0 = L["V"][tuple(Xe.astype(int).T)].astype(np.float64)
    sub = rng.choice(len(Xe), size=min(len(Xe), 10000), replace=False)
    Xs, ws, v0s = Xe[sub], we[sub], v0[sub]

    # ---- (1) convention search over the 24 mapping classes (centres D/2, D/2-0.5, and best offset)
    centres = [D / 2, D / 2 - 0.5] if o["centre"] == "auto" else [float(o["centre"])]
    conv = convention_search(L["V_coef"], Xe, v0, we, u_map, centres, Vlp=L["V"], m=env,
                             with_offset=o["conv_offset"])
    hyp = [r for r in conv if r["label"] == HYPOTHESIS_LABEL]
    best = conv[0]
    far = [r for r in conv if line_angle(r["axis_array"], best["axis_array"]) > 15.0]
    R["convention"] = dict(table=conv, best=best, hypothesis=hyp[0] if hyp else None,
                           contrast_vs_next_distinct=(best["corr"] - far[0]["corr"]) if far else None,
                           next_distinct=far[0] if far else None)
    cc = ", ".join(f"{k}: {v:.4f}" for k, v in best["corr_centre"].items())
    log(f"[convention] best mapping {best['label']} (== {best['alt']}): corr(V,CV) through centre {{{cc}}}"
        + (f", with best offset {best['corr_offset']:.4f} (axis offset {np.linalg.norm(best['T']) / 2 * apix:.2f} A)"
           if "corr_offset" in best else ""))
    if hyp:
        log(f"[convention] cryoDRGN hypothesis {HYPOTHESIS_LABEL}: corr {hyp[0]['corr']:.4f}; "
            f"{line_angle(best['axis_array'], hyp[0]['axis_array']):.2f} deg from best")
    if far:
        log(f"[convention] next distinct candidate (>15 deg away) {far[0]['label']} "
            f"corr = {far[0]['corr']:.4f}  (contrast {best['corr'] - far[0]['corr']:.4f})")

    if o["mapping"] != "auto":
        ua = parse_mapping(o["mapping"]) @ u_map
        nominal = TwoFold(ua, centres[0])
        log(f"[convention] mapping forced to {o['mapping']}")
    else:
        nominal = TwoFold(best["axis_array"], max(best["corr_centre"], key=best["corr_centre"].get))
    ops = {"nominal": nominal}
    # mapped u kept fixed, best perpendicular offset (FFT initialisation, then Nelder-Mead)
    T0 = best_offset_fft(L["V"], env, nominal.u, nominal.c)
    op_noff, _ = refine_twofold(L["V_coef"], Xs, v0s, ws, nominal.c, nominal.u, fit_axis=False,
                                fit_offset=True, T0=T0)
    ops["nominal+offset"] = op_noff
    R["selfcorr"] = {"nominal": selfcorr(L["V_coef"], Xe, v0, we, nominal, 3),
                     "nominal+offset": selfcorr(L["V_coef"], Xe, v0, we, op_noff, 3)}
    log(f"[convention] mapped u + best offset: axis offset {np.linalg.norm(op_noff.T) / 2 * apix:.2f} A, "
        f"corr {R['selfcorr']['nominal+offset']:.4f} (through centre {R['selfcorr']['nominal']:.4f})")

    # ---- (2) independent global search for the best 2-fold of V
    if o["global_search"]:
        t1 = time.time()
        # (a) real space, 2-fold through the box centre (sharp, but blind to an off-centre axis)
        Vc = sp.lowpass("V", maps["V"]["full"], o["coarse_lowpass"])
        subc = rng.choice(len(Xe), size=min(len(Xe), 8000), replace=False)
        dirs, corr, peaks = coarse_axis_search(Vc, Xe[subc], we[subc], D / 2, o["grid_deg"])
        # (b) offset-invariant self-rotation function (Patterson of the band-passed V)
        dirs2, corr2, peaks2 = self_rotation_search(sp, maps["V"]["full"], env, o["grid_deg"],
                                                    band=tuple(o["srf_band"]), rmax_A=o["srf_rmax"], rng=rng)
        pk = [dict(source="real-space", axis_array=dirs[i].tolist(), corr=float(corr[i]),
                   angle_to_nominal=line_angle(dirs[i], nominal.u)) for i in peaks[:6]]
        pk2 = [dict(source="self-rotation", axis_array=dirs2[i].tolist(), corr=float(corr2[i]),
                    angle_to_nominal=line_angle(dirs2[i], nominal.u)) for i in peaks2[:6]]
        # (c) multi-start refinement of axis + offset: mapped u and the top 3 peaks of each search
        starts = [("mapped u", nominal.u)] + [(p["source"], np.array(p["axis_array"])) for p in pk[:3] + pk2[:3]]
        uniq = []
        for src_, ax in starts:
            if all(line_angle(ax, a2) > 4.0 for _, a2 in uniq):
                uniq.append((src_, ax))
        cands = []
        for src_, ax in uniq:
            T0 = best_offset_fft(L["V"], env, ax, D / 2)
            op_c, _ = refine_twofold(L["V_coef"], Xs, v0s, ws, D / 2, ax, fit_axis=True, fit_offset=True, T0=T0)
            cands.append(dict(start=src_, corr=selfcorr(L["V_coef"], Xe, v0, we, op_c, 3), _op=op_c,
                              angle_to_nominal=line_angle(op_c.u, nominal.u), **op_c.describe(apix)))
        cands.sort(key=lambda d_: -d_["corr"])
        op_ref = cands[0]["_op"]
        ops["refined"] = op_ref
        R["selfcorr"]["refined"] = cands[0]["corr"]
        dev = line_angle(op_ref.u, nominal.u)
        R["global_search"] = dict(
            n_dirs=int(len(dirs)), grid_deg=o["grid_deg"], peaks_realspace=pk, peaks_selfrotation=pk2,
            candidates=cands, best_refined=op_ref.describe(apix), angle_refined_vs_nominal=dev,
            angle_top_realspace_vs_nominal=pk[0]["angle_to_nominal"],
            angle_top_selfrotation_vs_nominal=pk2[0]["angle_to_nominal"],
            angle_refined_vs_hypothesis=(line_angle(op_ref.u, hyp[0]["axis_array"]) if hyp else None),
            seconds=round(time.time() - t1, 1))
        R["_coarse"] = dict(dirs=dirs, corr=corr, dirs2=dirs2, corr2=corr2)
        log(f"[global] {len(dirs)} directions x 2 searches + {len(cands)} refinements in {time.time() - t1:.1f}s")
        log("[global] real-space peaks (corr, deg from mapped u): "
            + ", ".join(f"({p['corr']:.3f}, {p['angle_to_nominal']:.1f})" for p in pk[:4]))
        log("[global] self-rotation peaks (corr, deg from mapped u): "
            + ", ".join(f"({p['corr']:.3f}, {p['angle_to_nominal']:.1f})" for p in pk2[:4]))
        log("[global] refined candidates (start: corr, deg from mapped u, offset A): "
            + "; ".join(f"{c['start']}: {c['corr']:.4f}, {c['angle_to_nominal']:.1f}, {c['axis_offset_A']}"
                        for c in cands))
        log(f"[global] best 2-fold of V: axis {np.round(op_ref.u, 4).tolist()} (array order), {dev:.2f} deg from "
            f"mapped u, axis offset {np.linalg.norm(op_ref.T) / 2 * apix:.2f} A, corr {cands[0]['corr']:.4f}")
        if dev > o["max_axis_dev"]:
            log(f"WARNING: best 2-fold of V is {dev:.1f} deg from the mapped u -> check conventions / u.")
    op_name = o["operator"] if o["operator"] in ops else "nominal+offset"
    op = ops[op_name]
    R["operator_used"] = dict(name=op_name, **op.describe(apix))
    log(f"[operator] using '{op_name}': {op.describe(apix)}")
    if np.linalg.norm(op.T) / 2 * apix > 1.5:
        log(f"NOTE: the pseudo-2-fold passes {np.linalg.norm(op.T) / 2 * apix:.1f} A from the box centre; the swap "
            "operator then includes a translation. Cross-check with the pose shift differences (J487 - J486).")

    # ---- masks (envelope symmetrised under C; focused mask from V, C(V) at mask_lowpass)
    Vm_coef = sp.lowpass("V", maps["V"]["full"], o["mask_lowpass"], coeffs=True)[1]

    def masks_for(op_):
        CVm_ = op_.volume(Vm_coef)
        env_s_ = symmetrize_mask(env, op_)
        foc_ = symmetrize_mask(focus_mask(Vm, CVm_, env_s_, apix, level=o["focus_level"], excl=o["focus_excl"],
                                          dilate_A=o["focus_dilate"], soft_A=o["focus_soft"], background=bg), op_)
        return env_s_, foc_, CVm_

    env_s, foc, CVm = masks_for(op)
    masks = {"global": env_s, "focused": foc}
    R["mask"]["n_vox_global"] = int((env_s > 0.5).sum())
    R["mask"]["n_vox_focused"] = int((foc > 0.5).sum())
    log(f"[mask] global (C-symmetrised) {R['mask']['n_vox_global']} vox; focused {R['mask']['n_vox_focused']} vox "
        f"(level {o['focus_level']}, excl {o['focus_excl']})")
    if R["mask"]["n_vox_focused"] < 50:
        log("WARNING: focused mask is tiny; V and C(V) hardly differ -> p cannot be determined.")

    # ---- noise models for the bootstrap
    noise = None
    if o["n_boot"] and maps["V"].get("a") is not None:
        noise = {"V": NoiseModel(sp, maps["V"]["a"], maps["V"]["b"], o["lowpass"])}
        for M in ("M7", "M6"):
            if M in maps and maps[M].get("a") is not None:
                noise[M] = NoiseModel(sp, maps[M]["a"], maps[M]["b"], o["lowpass"])

    # ---- main fits
    fits, CV = evaluate_masks(L, op, masks, o, rng, o["n_boot"], sp, noise, log)
    R["fits"] = strip_private(fits)
    for mn in ("focused", "global"):
        f = fits[mn]
        s = f"[fit:{mn}] corr(V,CV)={f['corr_V_CV']:.3f} asym={f['asym_frac']:.3f}"
        for M in ("M7", "M6"):
            if M in f:
                r = f[M]
                s += (f" | {M}: p={r['p']:.3f} (a={r['alpha']:.3f}, b={r['beta']:.3f}, R2={r['r2']:.3f}"
                      f", SEhalf={r.get('se_half', float('nan')):.3f}, SEboot={r.get('se_boot', float('nan')):.3f})")
        log(s)
    if "M7" in fits["focused"] and "M6" in fits["focused"]:
        f7, f6 = fits["focused"]["M7"], fits["focused"]["M6"]
        p7, p6 = f7["p"], f6["p"]
        d7 = f7["antisym"]["diff"]
        # REVIEW FIX: alpha+beta and alpha-beta are in units of V's scale; V and M (different particle
        # subsets) need not share it.  Normalise by the scale of the symmetric dimer core, kappa.
        kap = core_scale(L, op, env_s, foc, "M7")
        rel = f7["sum"] / kap if kap > 0 else float("nan")
        drel = d7 / kap if kap > 0 else float("nan")
        # REVIEW ADDITION: V and M are made from the same number of particles, so the ratio of their
        # half-map noise RMS estimates the pure scale factor k.  Junk lowers the signal, not the noise.
        # Then (alpha+beta)/k ~ a+b (share of swapped particles with register-specific MPP6) and
        # kappa/k ~ a+b+s_dimer (share with a dimer core).
        def _noise_rms(d, key):
            if d.get("a") is None or d.get("b") is None:
                return None
            n = sp.lowpass(key, (d["a"] - d["b"]) / 2.0, o["lowpass"])
            return float(np.sqrt((n[env_s > 0.5] ** 2).mean()))

        nV, n7 = _noise_rms(maps["V"], "_noiseV"), _noise_rms(maps["M7"], "_noiseM7")
        kn = (n7 / nV) if (nV and n7) else None
        abs_reg = f7["sum"] / kn if kn else None
        abs_core = kap / kn if kn else None
        R["summary"] = dict(
            p_focused_M7=p7, p_focused_M6=p6, p7_plus_p6=p7 + p6, p_combined=(p7 + 1 - p6) / 2,
            se_boot_M7=f7.get("se_boot"), se_half_M7=f7.get("se_half"),
            boot_p16_p84_M7=f7.get("boot_p16_p84"),
            alpha_plus_beta_focused_M7=f7["sum"], alpha_minus_beta_M7=d7,
            # if V and M share an absolute scale (same particle number, cryoDRGN normalisation),
            # alpha - beta = a - b is a fraction of the swapped particles and a + b <= 1 bounds p:
            p_bounds_if_same_scale=sorted([0.5 + d7 / 2, 0.5 + math.copysign(0.5, d7)]),
            # scale-free versions (kappa = M/V scale on the dimer core):
            kappa_core_M7=kap, kappa_core_M6=core_scale(L, op, env_s, foc, "M6"),
            alpha_plus_beta_rel_M7=rel, alpha_minus_beta_rel_M7=drel,
            p_bounds_normalised=(sorted([0.5 + drel / 2, 0.5 + math.copysign(0.5, drel)])
                                 if math.isfinite(drel) else None),
            noise_ratio_M7_over_V=kn, register_share_abs_M7=abs_reg, dimer_share_abs_M7=abs_core,
            p_global_M7=fits["global"]["M7"]["p"], p_global_M6=fits["global"]["M6"]["p"],
            p_focused_M7_2reg=f7.get("two_reg", {}).get("p"), alpha_plus_beta_focused_M7_2reg=f7.get("two_reg", {}).get("sum"),
            delta_focused_M7=f7.get("delta"))
        log(f"[scale] M7/V scale on the dimer core kappa = {kap:.3f}; (alpha+beta)/kappa = {rel:.3f}, "
            f"(alpha-beta)/kappa = {drel:+.3f}")
        if not (0.8 < kap < 1.25):
            log(f"NOTE: M7 and V differ in scale on the dimer core (kappa = {kap:.2f}); read alpha+beta and the "
                "alpha-beta bound from their normalised (/kappa) versions.")
        if kn:
            log(f"[scale] half-map noise ratio M7/V = {kn:.3f} (pure scale factor if N and processing match): share of "
                f"swapped particles with register-specific density ~ {abs_reg:.3f}, with a dimer core ~ {abs_core:.3f}")
        weak = (not math.isfinite(rel) or rel < 0.25 or (f7.get("se_boot") or 0) > 0.1
                or (abs_reg is not None and abs_reg < 0.25))
        if weak:
            log(f"WARNING: register-specific amplitude is small: (alpha+beta)/kappa = {rel:.3f}"
                + (f", (alpha+beta)/noise-ratio = {abs_reg:.3f}" if abs_reg is not None else "")
                + f" (raw alpha+beta {f7['sum']:.3f}, SE_boot {f7.get('se_boot') or float('nan'):.3f}): little register"
                " information; p is poorly constrained / undefined.")
        if kap < 0.25 or (abs_core is not None and abs_core < 0.5):
            log(f"WARNING: M7 has little dimer-core density relative to V (kappa = {kap:.3f}"
                + (f", kappa/noise-ratio = {abs_core:.3f}" if abs_core is not None else "")
                + "): mostly junk/empty particles, or a large scale mismatch.")

    # ---- operator sensitivity (same masks recomputed per operator; no bootstrap)
    R["operator_sensitivity"] = {}
    for name, opx in ops.items():
        if name == op_name:
            continue
        e2, f2, _ = masks_for(opx)
        fx, _ = evaluate_masks(L, opx, {"global": e2, "focused": f2}, o, rng, 0, log=log)
        R["operator_sensitivity"][name] = {mn: {M: fx[mn][M]["p"] for M in ("M7", "M6") if M in fx[mn]}
                                           for mn in fx}

    # ---- low-pass / focus-level sweep (half-map SE only)
    R["sweep"] = []
    for res in o["sweep_lowpass"] or []:
        Lx = L if res == o["lowpass"] else lowpass_all(res)
        fx, _ = evaluate_masks(Lx, op, masks, o, rng, 0, log=log)
        for mn in fx:
            for M in ("M7", "M6"):
                if M in fx[mn]:
                    R["sweep"].append(dict(kind="lowpass", lowpass=res, focus_level=o["focus_level"],
                                           focus_excl=o["focus_excl"], mask=mn, map=M, p=fx[mn][M]["p"],
                                           sum=fx[mn][M]["sum"], se_half=fx[mn][M].get("se_half")))
    for lev, exc in o["sweep_focus"] or []:
        f2 = symmetrize_mask(focus_mask(Vm, CVm, env_s, apix, level=lev, excl=exc, dilate_A=o["focus_dilate"],
                                        soft_A=o["focus_soft"], background=bg), op)
        if (f2 > 0.5).sum() < 20:
            continue
        fx, _ = evaluate_masks(L, op, {"focused": f2}, o, rng, 0, log=log)
        for M in ("M7", "M6"):
            if M in fx["focused"]:
                R["sweep"].append(dict(kind="focus", lowpass=o["lowpass"], focus_level=lev, focus_excl=exc,
                                       mask="focused", map=M, p=fx["focused"][M]["p"],
                                       sum=fx["focused"][M]["sum"], se_half=fx["focused"][M].get("se_half"),
                                       n_vox=int((f2 > 0.5).sum())))

    # ---- diagnostics: is M6 just C(M7)?
    A = dict(V=L["V"], CV=CV, env=env_s, foc=foc)
    if o["diagnostics"] and "M7" in L and "M6" in L:
        idx = fits["global"]["_idx"]
        X, w = fits["global"]["_X"], fits["global"]["_w"]
        m6 = L["M6"].ravel()[idx]
        cm7 = op.sample(L["M7_coef"], X)
        R["diagnostics"] = dict(corr_M6_CM7=wcorr(m6, cm7, w), corr_M6_M7=wcorr(m6, L["M7"].ravel()[idx], w),
                                corr_M7_V=wcorr(L["M7"].ravel()[idx], fits["global"]["_v"], w),
                                corr_M7_CV=wcorr(L["M7"].ravel()[idx], fits["global"]["_cv"], w))
        log(f"[diag] corr(M6, C M7) = {R['diagnostics']['corr_M6_CM7']:.4f} vs corr(M6, M7) = "
            f"{R['diagnostics']['corr_M6_M7']:.4f}  (~1 => M6 is a rotated copy of M7, not a replicate)")
        A["M7"], A["M6"] = L["M7"], L["M6"]
        A["CM7"], A["CM6"] = op.volume(L["M7_coef"]), op.volume(L["M6_coef"])
    R["seconds"] = round(time.time() - t0, 1)
    A["_coarse"] = R.pop("_coarse", None)
    return R, A


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bgr", ["#1f5fae", "#7fa9df", "#e6e5e1", "#e58a89", "#b8302f"])


def make_figures(R, A, apix, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    files = []
    V, CV, env, foc = A["V"], A["CV"], A["env"], A["foc"]
    dif = V - CV
    D = V.shape[0]
    pk = np.unravel_index(np.argmax(np.abs(dif) * (env > 0.5)), dif.shape)
    rows = [("V", V, "gray"), ("C(V)", CV, "gray"), ("V - C(V)", dif, "div")]
    if "M7" in A:
        rows += [("M7 - C(M7)", A["M7"] - A["CM7"], "div"), ("M6 - C(M6)", A["M6"] - A["CM6"], "div")]
    vmax_d = float(np.percentile(np.abs(dif[env > 0.5]), 99.8))
    vlo, vhi = np.percentile(V[env > 0.5], [1, 99.8])
    nz = np.argwhere(env > 0.05)  # crop to the mask bounding box (+ margin)
    blo = np.maximum(nz.min(0) - 6, 0)
    bhi = np.minimum(nz.max(0) + 7, D)
    fig, axs = plt.subplots(len(rows), 3, figsize=(10.5, 3.1 * len(rows)), constrained_layout=True)
    axn = ["axis 0 (z)", "axis 1 (y)", "axis 2 (x)"]
    for r, (name, vol, kind) in enumerate(rows):
        for c in range(3):
            sl = [slice(int(blo[k]), int(bhi[k])) for k in range(3)]
            sl[c] = pk[c]
            others = [k for k in range(3) if k != c]
            ext = ((blo[others[1]] - D / 2) * apix, (bhi[others[1]] - D / 2) * apix,
                   (blo[others[0]] - D / 2) * apix, (bhi[others[0]] - D / 2) * apix)
            img = vol[tuple(sl)]
            ax = axs[r, c]
            if kind == "gray":
                im = ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, origin="lower", extent=ext)
            else:
                im = ax.imshow(img, cmap=_div_cmap(), vmin=-vmax_d, vmax=vmax_d, origin="lower", extent=ext)
            ax.contour(env[tuple(sl)], levels=[0.5], colors=[GRAY], linewidths=0.8, origin="lower", extent=ext)
            ax.contour(foc[tuple(sl)], levels=[0.5], colors=["#c98500"], linewidths=0.9, linestyles="--",
                       origin="lower", extent=ext)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(f"slice {axn[c]} = {pk[c]}  (A from centre)", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
        fig.colorbar(im, ax=axs[r, :], shrink=0.8)
    fig.suptitle("Slices through max |V - C(V)| (MPP6 region). Gray contour: global mask; "
                 "dashed: focused mask. Difference rows share one color scale.", fontsize=9)
    f = os.path.join(outdir, "fig_slices.png")
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(f)

    # ---- fit figure
    fits = R["fits"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    idx = np.flatnonzero(foc.ravel() > 0.5)
    xa = dif.ravel()[idx]
    for k, (M, col) in enumerate((("M7", BLUE), ("M6", ORANGE))):
        ax = axs[0, k]
        if M not in A:
            ax.axis("off")
            continue
        ya = (A[M] - A["C" + M]).ravel()[idx]
        ax.hexbin(xa, ya, gridsize=60, cmap="Greys", mincnt=1, bins="log")
        d = fits["focused"][M]["alpha"] - fits["focused"][M]["beta"]
        xx = np.linspace(xa.min(), xa.max(), 10)
        ax.plot(xx, d * xx, color=col, lw=2, label=f"slope alpha-beta = {d:.3f}")
        ax.plot(xx, xx, color=GRAY, lw=1, ls=":", label="slope 1 (pure concordant)")
        ax.plot(xx, -xx, color=GRAY, lw=1, ls="--", label="slope -1 (pure swapped)")
        ax.set_xlabel("V - C(V)   (focused mask voxels)")
        ax.set_ylabel(f"{M} - C({M})")
        ax.set_title(f"{M}: antisymmetric parts  (p = {fits['focused'][M]['p']:.3f})", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    # p summary
    ax = axs[1, 0]
    ylab, y = [], 0
    for mn in ("focused", "global"):
        for M, col in (("M7", BLUE), ("M6", ORANGE)):
            if M not in fits[mn]:
                continue
            r = fits[mn][M]
            pv = r["p"] if M == "M7" else 1 - r["p"]
            se = r.get("se_boot") or r.get("se_half") or 0
            ax.errorbar(pv, y, xerr=se, fmt="o", color=col, ms=8, capsize=4, lw=2)
            for ph in r.get("p_halfM", []):
                ax.plot(ph if M == "M7" else 1 - ph, y, "|", color=col, ms=14, mew=1.5)
            ylab.append(f"{mn}: {'p(M7)' if M == 'M7' else '1 - p(M6)'}")
            y += 1
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab, fontsize=9)
    ax.axvline(0.5, color=GRAY, lw=1, ls=":")
    ax.set_xlabel("p = a / (a + b)   (dot: full map +/- bootstrap SE; ticks: half maps)")
    ax.set_title("Register fraction estimates", fontsize=10)
    ax.grid(axis="x", color="#e6e5e1")
    # sweep
    ax = axs[1, 1]
    sw = [s for s in R.get("sweep", []) if s["kind"] == "lowpass" and s["map"] == "M7"]
    for mn, col, mk in (("focused", BLUE, "o"), ("global", AQUA, "s")):
        pts = sorted([(s["lowpass"], s["p"], s["se_half"] or 0) for s in sw if s["mask"] == mn])
        if pts:
            xs, ps, es = zip(*pts)
            ax.errorbar(xs, ps, yerr=es, color=col, marker=mk, ms=8, capsize=4, lw=2, label=f"{mn} mask")
    ax.set_xlabel("low-pass (A)")
    ax.set_ylabel("p(M7)")
    ax.set_title("Sensitivity of p(M7) to low-pass and mask (error bars: half-map SE)", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(color="#e6e5e1")
    f = os.path.join(outdir, "fig_fit.png")
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(f)

    # ---- axis figure
    conv = R["convention"]["table"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    ax = axs[0]
    cols = [BLUE if r["label"] == R["convention"]["best"]["label"] else
            (ORANGE if r["label"] == HYPOTHESIS_LABEL else "#b9b8b2") for r in conv]
    ax.barh(range(len(conv)), [r["corr"] for r in conv], color=cols, height=0.7)
    ax.set_yticks(range(len(conv)))
    ax.set_yticklabels([r["label"] for r in conv], fontsize=7, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("corr(V, C(V)) in envelope mask (best centre/offset)")
    ax.set_title("24 axis mappings of u; blue = chosen, orange = cryoDRGN [z,y,x]", fontsize=9)
    lo = min(r["corr"] for r in conv)
    ax.set_xlim(min(0, lo) - 0.02, 1.0)
    co = A.get("_coarse")

    def lam(v):  # Lambert azimuthal equal-area projection of axis lines about array axis 0
        v = np.atleast_2d(np.asarray(v, float))
        v = v * np.where(v[:, :1] < 0, -1.0, 1.0)
        k = np.sqrt(2 / (1 + v[:, 0]))
        return k * v[:, 2], k * v[:, 1]

    for j, (dk, ck, title) in enumerate((("dirs", "corr", "real space, 2-fold through box centre"),
                                         ("dirs2", "corr2", "self-rotation (offset-invariant)"))):
        ax = axs[1 + j]
        if co is None:
            ax.axis("off")
            continue
        x, y = lam(co[dk])
        sc = ax.scatter(x, y, c=co[ck], s=6, cmap="viridis")
        fig.colorbar(sc, ax=ax, shrink=0.85, label="correlation")
        for vec, mk, lab in ((R["convention"]["best"]["axis_array"], "o", "u mapped (chosen)"),
                             (R["global_search"]["best_refined"]["axis_array"], "x", "best 2-fold (refined)")):
            px, py = lam(vec)
            ax.plot(px, py, mk, ms=13, mfc="none", mec="#e34948", mew=2, label=lab)
        ax.set_aspect("equal")
        ax.set_xlabel("axis 2 (x) component, equal-area")
        ax.set_ylabel("axis 1 (y) component, equal-area")
        ax.set_title(f"Global 2-fold search: {title}", fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
    f = os.path.join(outdir, "fig_axis.png")
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(f)
    return files


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def format_report(R):
    L = []
    o = R["inputs"]
    nan = float("nan")
    L.append("REGISTER FRACTION REPORT")
    L.append(f"shape {o['shape']}  apix {o['apix']}  u(map xyz) {np.round(o['u_map_xyz'], 4).tolist()}  "
             f"low-pass {o['options']['lowpass']} A  runtime {R['seconds']} s")
    c = R["convention"]
    L.append("")
    L.append("1. Axis convention (array_vec = P @ map_vec; a class {P, -P} defines one 2-fold axis)")
    b = c["best"]
    L.append(f"   chosen mapping {b['label']} (== {b['alt']}); corr(V,CV) through centre "
             + ", ".join(f"{k}: {v:.4f}" for k, v in b["corr_centre"].items())
             + (f"; with best offset {b['corr_offset']:.4f}" if "corr_offset" in b else ""))
    if c["hypothesis"]:
        L.append(f"   cryoDRGN hypothesis {HYPOTHESIS_LABEL}: corr {c['hypothesis']['corr']:.4f}, "
                 f"{line_angle(b['axis_array'], c['hypothesis']['axis_array']):.2f} deg from chosen")
    if c["next_distinct"]:
        L.append(f"   best candidate >15 deg away: {c['next_distinct']['label']} corr {c['next_distinct']['corr']:.4f}"
                 f"  -> contrast {c['contrast_vs_next_distinct']:.4f}")
    L.append("   top 6 mapping classes (score, deg from chosen):")
    for r in c["table"][:6]:
        L.append(f"     {r['label']:>12s} corr {r['corr']:.4f}  {line_angle(r['axis_array'], b['axis_array']):5.1f} deg")
    if "global_search" in R:
        g = R["global_search"]
        L.append(f"   global search: best 2-fold of V {np.round(g['best_refined']['axis_array'], 4).tolist()} (array order)"
                 f" = {g['angle_refined_vs_nominal']:.2f} deg from mapped u; axis offset {g['best_refined']['axis_offset_A']} A")
        L.append(f"   top coarse peak: real-space {g['angle_top_realspace_vs_nominal']:.1f} deg, self-rotation "
                 f"{g['angle_top_selfrotation_vs_nominal']:.1f} deg from mapped u")
        L.append("   real-space peaks (corr, deg): " + ", ".join(
            f"({p['corr']:.3f}, {p['angle_to_nominal']:.1f})" for p in g["peaks_realspace"][:5]))
        L.append("   self-rotation peaks (corr, deg): " + ", ".join(
            f"({p['corr']:.3f}, {p['angle_to_nominal']:.1f})" for p in g["peaks_selfrotation"][:5]))
    L.append("   self-correlations: " + ", ".join(f"{k} {v:.4f}" for k, v in R["selfcorr"].items()))
    L.append(f"   operator used: {R['operator_used']['name']}  axis {R['operator_used']['axis_array']}  "
             f"offset {R['operator_used']['axis_offset_A']} A")
    L.append("")
    L.append("2. Fits  M ~ alpha V + beta C(V)" + (" + delta min(V, C(V))" if o["options"].get("common_regressor", True)
                                                     else "") + " + gamma ,  p = alpha / (alpha + beta)")
    for mn in ("focused", "global"):
        f = R["fits"][mn]
        L.append(f"   [{mn} mask] {f['n_vox_support']} voxels in support, corr(V,CV) {f['corr_V_CV']:.3f}, "
                 f"asymmetry |V-CV|/|V+CV| {f['asym_frac']:.3f}")
        for M in ("M7", "M6"):
            if M not in f:
                continue
            r = f[M]
            L.append(f"     {M}: p = {r['p']:.4f}  alpha {r['alpha']:.4f} beta {r['beta']:.4f} "
                     f"(alpha+beta {r['sum']:.3f}, alpha-beta {r['antisym']['diff']:+.3f}) R2 {r['r2']:.3f}")
            L.append(f"         antisym-only p {r['antisym']['p']:.4f}; half M {np.round(r.get('p_halfM', []), 4).tolist()}"
                     f" half V {np.round(r.get('p_halfV', []), 4).tolist()} SE_half {r.get('se_half', nan):.4f}"
                     f" SE_boot {r.get('se_boot', nan):.4f} boot16-84 {np.round(r.get('boot_p16_p84', []), 4).tolist()}")
            if "two_reg" in r:
                L.append(f"         delta (MPP6-less symmetric content) {r['delta']:+.4f}; 2-regressor fit (no min(V,CV)): "
                         f"p {r['two_reg']['p']:.4f}, alpha+beta {r['two_reg']['sum']:.3f}")
    if "summary" in R:
        s = R["summary"]
        L.append(f"   HEADLINE (focused mask): p(M7) = {s['p_focused_M7']:.3f} +/- {s['se_boot_M7'] or nan:.3f} (stat.),"
                 f" p(M6) = {s['p_focused_M6']:.3f}, p7+p6 = {s['p7_plus_p6']:.3f}")
        L.append(f"   raw alpha+beta (focused) = {s['alpha_plus_beta_focused_M7']:.3f}, alpha-beta = "
                 f"{s['alpha_minus_beta_M7']:+.3f} (units of V's scale; bound IF V and M share a scale: p in "
                 f"{np.round(s['p_bounds_if_same_scale'], 3).tolist()})")
        if "kappa_core_M7" in s:
            pbn = s.get("p_bounds_normalised")
            L.append(f"   M/V scale on the C-symmetric dimer core: kappa = {s['kappa_core_M7']:.3f} (M7), "
                     f"{s['kappa_core_M6']:.3f} (M6)")
            L.append(f"   scale-free: (alpha+beta)/kappa = {s['alpha_plus_beta_rel_M7']:.3f} (~1: every dimer-bearing "
                     f"swapped particle has register-specific MPP6; ~2 at p~0.5: MPP6 on both protomers (2:2); "
                     f"<0.25: little register information), (alpha-beta)/kappa = {s['alpha_minus_beta_rel_M7']:+.3f}"
                     + (f" -> p in {np.round(pbn, 3).tolist()} for any amount of symmetric content" if pbn else ""))
        if s.get("noise_ratio_M7_over_V"):
            L.append(f"   half-map noise ratio M7/V = {s['noise_ratio_M7_over_V']:.3f} (pure scale, same N): share of swapped "
                     f"particles with register-specific density ~ {s['register_share_abs_M7']:.3f}, with a dimer core ~ "
                     f"{s['dimer_share_abs_M7']:.3f} (junk lowers both)")
        L.append(f"   global mask: p(M7) = {s['p_global_M7']:.3f}, p(M6) = {s['p_global_M6']:.3f} "
                 f"(pulled toward 0.5 by C-symmetric content the regressors cannot absorb)")
        if s.get("p_focused_M7_2reg") is not None:
            L.append(f"   2-regressor focused fit (original proposal): p(M7) = {s['p_focused_M7_2reg']:.3f}, alpha+beta "
                     f"{s['alpha_plus_beta_focused_M7_2reg']:.3f}; a gap to the headline means MPP6-less symmetric "
                     f"particles (delta = {s['delta_focused_M7']:+.3f}) leak into the focused mask")
    if R.get("operator_sensitivity"):
        L.append("   operator sensitivity (p for M7 / M6):")
        for name, d in R["operator_sensitivity"].items():
            L.append("     " + name + ": " + "; ".join(f"{mn} {v.get('M7', nan):.4f}/{v.get('M6', nan):.4f}"
                                                      for mn, v in d.items()))
    if R.get("sweep"):
        L.append("")
        L.append("3. Sensitivity sweep (p for M7 +/- half-map SE; alpha+beta)")
        for s in R["sweep"]:
            if s["map"] != "M7":
                continue
            L.append(f"     {s['kind']:8s} lowpass {s['lowpass']:5.1f} level {s['focus_level']:.2f} excl "
                     f"{s['focus_excl']:.2f} {s['mask']:8s} p = {s['p']:.4f} +/- "
                     f"{s['se_half'] if s['se_half'] is not None else nan:.4f}  alpha+beta {s['sum']:.3f}"
                     + (f"  ({s['n_vox']} vox)" if "n_vox" in s else ""))
    if "diagnostics" in R:
        d = R["diagnostics"]
        L.append("")
        L.append(f"4. Diagnostics: corr(M6, C(M7)) = {d['corr_M6_CM7']:.4f}, corr(M6, M7) = {d['corr_M6_M7']:.4f}, "
                 f"corr(M7, V) = {d['corr_M7_V']:.4f}, corr(M7, CV) = {d['corr_M7_CV']:.4f}")
        L.append("   If corr(M6, C(M7)) ~ 1, M6 is a rotated copy of M7 and p6 = 1 - p7 is NOT independent evidence.")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vdir", required=True, help="concordant backprojection folder (reference register)")
    ap.add_argument("--m7dir", required=True, help="swapped particles, J487 poses")
    ap.add_argument("--m6dir", required=True, help="swapped particles, J486 poses")
    ap.add_argument("--axis", type=float, nargs=3, required=True, metavar=("UX", "UY", "UZ"),
                    help="swap axis u in MAP (x,y,z) coordinates (axis of R_J487^T R_J486, cryoDRGN rot)")
    ap.add_argument("--apix", type=float, default=None, help="pixel size (default: from MRC header)")
    ap.add_argument("--lowpass", type=float, default=DEFAULTS["lowpass"], help="analysis low-pass (A)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-lowpass", type=float, default=DEFAULTS["mask_lowpass"])
    ap.add_argument("--mask-kda", type=float, default=DEFAULTS["mask_kda"],
                    help="threshold V so the binary mask encloses this mass (kDa, 1.21 A^3/Da)")
    ap.add_argument("--mask-threshold", type=float, default=None, help="absolute threshold (overrides --mask-kda)")
    ap.add_argument("--mask-dilate", type=float, default=DEFAULTS["mask_dilate"], help="A")
    ap.add_argument("--mask-soft", type=float, default=DEFAULTS["mask_soft"], help="A")
    ap.add_argument("--focus-level", type=float, default=DEFAULTS["focus_level"],
                    help="focused mask: |V-CV| > level * 99.5th percentile")
    ap.add_argument("--focus-excl", type=float, default=DEFAULTS["focus_excl"],
                    help="focused mask: |V-CV| > excl * max(V,CV) (exclusive density; 0 disables)")
    ap.add_argument("--centre", default="auto", help="'auto' (test D/2 and D/2-0.5) or a number")
    ap.add_argument("--mapping", default="auto", help="'auto' or a mapping label like '[+z,+y,+x]'")
    ap.add_argument("--operator", default="refined", choices=("refined", "nominal", "nominal+offset"))
    ap.add_argument("--no-global-search", action="store_true")
    ap.add_argument("--no-common-regressor", action="store_true",
                    help="fit only M ~ alpha V + beta C(V) + gamma (the original proposal); by default a third, "
                         "C-symmetric regressor min(V, C(V)) absorbs MPP6-less symmetric particles")
    ap.add_argument("--grid-deg", type=float, default=DEFAULTS["grid_deg"])
    ap.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-maps", action="store_true",
                    help="do not write maps/ (low-passed V, C(V), differences, masks) for ChimeraX")
    a = ap.parse_args(argv)

    maps, infos = {}, {}
    for k, d in (("V", a.vdir), ("M7", a.m7dir), ("M6", a.m6dir)):
        maps[k], infos[k] = load_dir(d, need_halves=False)
    apix = a.apix or infos["V"]["apix"]
    if apix is None:
        sys.exit("pixel size unknown: pass --apix")
    for k, inf in infos.items():
        if inf["axes"] != (1, 2, 3):
            print(f"WARNING: {k} has mapc/mapr/maps = {inf['axes']} (not 1,2,3); the convention search "
                  "will still find the axis, but the array is not [z,y,x].")
        if inf["apix"] and abs(inf["apix"] - apix) > 1e-3:
            print(f"WARNING: {k} header apix {inf['apix']} differs from {apix}")
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    opts = dict(lowpass=a.lowpass, mask_lowpass=a.mask_lowpass, mask_kda=a.mask_kda,
                mask_threshold=a.mask_threshold, mask_dilate=a.mask_dilate, mask_soft=a.mask_soft,
                focus_level=a.focus_level, focus_excl=a.focus_excl, centre=a.centre, mapping=a.mapping, operator=a.operator,
                global_search=not a.no_global_search, grid_deg=a.grid_deg, n_boot=a.n_boot, seed=a.seed,
                common_regressor=not a.no_common_regressor)
    R, A = run_analysis(maps, apix, a.axis, opts, log=log)
    R["inputs"]["dirs"] = dict(V=os.path.abspath(a.vdir), M7=os.path.abspath(a.m7dir), M6=os.path.abspath(a.m6dir))
    rep = format_report(R)
    log("")
    log(rep)
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(rep + "\n")
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(strip_private(R), f, indent=1)
    if not a.no_figures:
        files = make_figures(R, A, apix, a.out)
        log("figures: " + ", ".join(files))
    if not a.no_maps:  # same [z,y,x] array convention and origin 0 as cryoDRGN's backproject.mrc
        md = os.path.join(a.out, "maps")
        os.makedirs(md, exist_ok=True)
        out_maps = {"V_lp": A["V"], "CV_lp": A["CV"], "V_minus_CV": A["V"] - A["CV"],
                    "mask_global": A["env"], "mask_focused": A["foc"]}
        if "M7" in A:
            out_maps.update({"M7_minus_CM7": A["M7"] - A["CM7"], "M6_minus_CM6": A["M6"] - A["CM6"]})
        for k, arr in out_maps.items():
            write_mrc(os.path.join(md, k + ".mrc"), arr, apix)
        log(f"maps: {md}/ ({', '.join(out_maps)})")
    logf.close()


if __name__ == "__main__":
    main()
