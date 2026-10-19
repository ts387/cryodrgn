## Task: how do the two refinements split the pseudo-C2-swapped particles? (MINA53–MPP6 cryo-EM)

You are helping me analyse three cryo-EM reconstructions in the folder I have given you access to. This is a
quantitative question where an axis-convention mistake gives a plausible-looking wrong answer, so follow the
checks below and do not quote a number that fails them. Treat every input file as read-only. Write everything
you produce to a new folder `regfrac_results/` inside the working folder.

### Background (you have no other context)
- Sample: MINA53, a JmjC protein that forms a dimer, bound to MPP6. The dimer is pseudo-2-fold symmetric and
  MPP6 breaks the symmetry. It is not settled whether there is one MPP6 per dimer or two.
- Two cryoSPARC v5 refinements of the same 418,418 particles disagree on orientation for some particles. The
  first is J485, whose poses the motion-correction job J486 passes on unchanged, so I call them "J486 poses".
  The second is J487. I matched particles by uid and computed each particle's relative rotation R_J487ᵀ·R_J486,
  using cryoDRGN's convention (`rot` maps map → image coordinates):
  - concordant (agree within 20°): 70.9%
  - c2_swap (~178° about one fixed map-frame axis): 8.0%, ~95% of them genuine 2-fold swaps
  - discordant (anything else, mostly poses that look unrelated): 21.1%
- The swap axis in map coordinates (x, y, z) is u = (+0.720, −0.692, +0.061). It is also stored in
  `pc2_summary.json` under `axis → body_modes[0] → axis`.
- For any one swapped particle, J487 may be right, J486 may be right, neither may be right, or the particle may
  itself be symmetric (MPP6 on both protomers, or on neither), in which case the swap is harmless. So the
  question is not "which job is right". It is: what fraction p of the register-informative swapped particles
  does J487 put in the same register as the concordant particles? J486 then has 1 − p.
- To measure this I made three reconstructions with `cryodrgn backproject_voxel` (cryoDRGN 4.3.1), all from the
  same box-128 particle stack (2.609375 Å/px), each from 32,638 particles and each with half maps:
  - V: concordant particles (a random 32,638 of them). This defines the reference register.
  - M7: the swapped particles with J487 poses.
  - M6: the same swapped particles with J486 poses.
- Model: M7 ≈ a·V + b·C(V) + (C-symmetric content) + noise, and M6 ≈ b·V + a·C(V) + …. Here C is the 180°
  rotation about u through (or near) the box centre, and p = a / (a + b).

### Inputs in the working folder
- Three folders whose names start with `concordant`, `c2swap_J487` and `c2swap_J486` (the concordant one is
  probably `concordant_eq32638`). Each holds `backproject.mrc`, `half_map_a.mrc`, `half_map_b.mrc`, and possibly
  `fsc-vals.txt` / `fsc-plot.png`.
- `pc2_summary.json`.
- `register_fraction.py` and `test_register_fraction.py`: a validated analysis script and its synthetic test
  suite. If they are not in the folder, download them:
  - https://raw.githubusercontent.com/ts387/cryodrgn/claude/cryodrgn-colab-notebook-5mf3p3/analysis_scripts/pseudo_c2/register_fraction.py
  - https://raw.githubusercontent.com/ts387/cryodrgn/claude/cryodrgn-colab-notebook-5mf3p3/analysis_scripts/pseudo_c2/test_register_fraction.py

  If you cannot download them, stop and ask me to add them. Do not write your own replacement. The script
  exists because the obvious estimators fail in specific ways (described in its docstring).

### What `register_fraction.py` does
Read its docstring fully before running it. In short:
- It low-passes all maps identically (8 Å by default) and builds two masks:
  - an envelope mask from V;
  - a focused mask on MPP6 and its symmetry mate: density that is in V but not in C(V), or the reverse.
- In the focused mask it fits M ≈ α·V + β·C(V) + δ·min(V, C(V)) + γ by weighted least squares, for M7 and M6,
  and reports p = α / (α + β).
  - The δ term absorbs density both registers share, such as MPP6-free dimers. Without it, p is pulled
    towards 0.5.
- It does not trust the axis convention:
  - It tests all 24 signed-permutation mappings of u into array order.
  - It searches independently for V's own best 2-fold axis.
  - It refines a small offset of the axis from the box centre.
- It reports:
  - statistical errors from half-map refits and a noise bootstrap;
  - scale-free signal measures: (α+β)/κ, and the half-map noise ratio with the derived share of swapped particles;
  - a sensitivity sweep over low-pass and focused-mask settings;
  - an operator comparison (refined, nominal and nominal+offset).
- It writes `report.txt`, `results.json`, `log.txt`, `fig_slices.png`, `fig_fit.png`, `fig_axis.png`, and
  diagnostic maps in `maps/`.

Verified conventions, from the cryoDRGN 4.3.1 source: `backproject.mrc` arrays are indexed [z, y, x] in the
same frame as u, with the origin at index 64 (D/2) and no mirroring. So u maps to array order
(u_z, u_y, u_x), which the report calls mapping `[+z,+y,+x]`.

### Steps
1. **Environment.** Python 3 with numpy, scipy and matplotlib; mrcfile is optional. Install them with pip if
   missing. Run the scripts with `python -I` from outside the data folders.
2. **Inspect the inputs without modifying them.**
   - For each of the nine MRC files, report the shape, the voxel size from the header, and min/mean/max.
   - All should be 128³ at 2.609375 Å/px. If they are all a different consistent size, use the header value
     and say so.
   - Confirm that `pc2_summary.json` gives the same axis as above.
   - If `fsc-vals.txt` files are present, report each map's FSC = 0.143 resolution.
3. **Self-test (about 15 minutes).**
   - Run `python -I test_register_fraction.py regfrac_results/selftest`.
   - The final SUMMARY line must show PASS for every test. If any test fails, stop and show me the failing
     lines. Do not run the real analysis.
4. **Run the analysis.**
   - Run `python -I register_fraction.py --vdir <concordant folder> --m7dir <c2swap_J487 folder> --m6dir
     <c2swap_J486 folder> --axis 0.720 -0.692 0.061 --lowpass 8 --out regfrac_results/main`.
   - Leave out `--apix` so the header value is used.
5. **Check before you interpret.** Use `report.txt` and the WARNING/NOTE lines in `log.txt`. Every check must
   pass before you quote p. If one fails, say which one and what that means, and stop interpreting.
   - **(a) Axis convention.**
     - The chosen mapping is `[+z,+y,+x]` or one of its near-degenerate neighbours: `[+z,-x,-y]` (about 2.3°
       away) or `[+z,-y,-x]` / `[-z,+y,+x]` (about 7° away).
     - The "contrast" to the best class more than 15° away is at least about 0.05.
     - If a class far from u wins, the axis is wrong. Stop.
   - **(b) Independent 2-fold.** V's best 2-fold from the global search is within about 5° of the mapped u
     (6° is the hard limit).
   - **(c) Axis offset.** It is at most about 3 Å. About 2.2 Å just reflects a half-voxel centre convention and
     is harmless.
   - **(d) Focused mask.**
     - It holds hundreds to a few thousand voxels.
     - corr(V, CV) is about 0.8 or higher in the global (envelope) mask, and low or negative in the focused mask.
     - In ChimeraX or in slices, the focused mask sits on MPP6 and its symmetry mate.
   - **(e) Consistency.**
     - p(M7) + p(M6) = 1 ± 0.03, and corr(M6, C(M7)) ≥ 0.98.
     - M6 is nearly a rotated copy of M7, so this checks the operator only. It is not independent evidence.
   - **(f) Fit sanity.**
     - 0 ≤ p ≤ 1 (α, β ≥ 0).
     - Focused-mask R² is at least about 0.8.
     - The "antisym-only p" is within about 0.02 of the fitted p.
     - p > 1 or β < 0 means V itself contains swapped particles. Report it.
   - **(g) Signal.**
     - (α+β)/κ ≥ 0.25.
     - The share of swapped particles with register-specific density is ≥ 0.25.
     - SE_boot ≤ 0.1.
     - There are no WARNING lines.
     - κ and the half-map noise ratio M7/V should both be about 1.
   - **(h) Stability.** The spread of the focused p across the section-3 sweep and the operator comparison is
     at most about 0.03–0.05.
   - **(i) Errors.** SE_half and SE_boot are of similar size.
6. **Result.** Quote p = (p(M7) + 1 − p(M6)) / 2. Give the statistical error from SE_boot, and the systematic
   error from the spread in check (h).
7. **Visual check.**
   - Write a ChimeraX command file, `regfrac_results/view.cxc`. It should open, at matched thresholds:
     - `maps/V_lp.mrc` and `maps/CV_lp.mrc`;
     - low-passed copies of M7 and M6, which you make with the same 8 Å filter (save them in
       `regfrac_results/`);
     - `maps/V_minus_CV.mrc` and `maps/M7_minus_CM7.mrc`;
     - `maps/mask_focused.mrc`.

     That way I can see MPP6 and the empty symmetry-mate site.
   - Also make one figure of matching slices through the MPP6 site for V, C(V), M7 and M6.

### Interpretation
In this table, p is the combined focused p, r = (α+β)/κ, and A is the share of swapped particles with
register-specific density.

| Result | Meaning | What I should do |
|---|---|---|
| p ≳ 0.9, r ≳ 0.5, A ≳ 0.25 | J487 puts the swapped particles in the concordant register; J486 has them flipped | Use J487 poses for them |
| p ≲ 0.1 (same signal conditions) | J486 is right; J487 has them flipped | Use J486 poses |
| 0.1 < p < 0.9, not ≈ 0.5 | Mixed: a fraction p is right in J487 and 1 − p in J486 | Assign the register per particle: C2 symmetry expansion plus focused 3D classification without alignment on MPP6, or symmetry relaxation in refinement |
| p ≈ 0.5, r ≈ 1 | A genuine 50:50 register mix | Per-particle classification |
| p ≈ 0.5, r ≈ 2 | MPP6 on both protomers (2:2), so the swap is immaterial. This assumes V's MPP6 occupancy matches theirs | Treat them as 2:2, not as errors |
| r < 0.25, A < 0.25, or SE_boot > 0.1 | p is undefined: these particles carry no MPP6/register signal (MPP6-free dimers or junk). κ / noise ratio < 0.5 means junk dominates | Neither job is "wrong" for them |

State these caveats in the report:
- p is relative to the concordant set's register. If the concordant set contains a few swapped particles itself,
  p is pushed away from 0.5.
- The map analysis gives a population fraction, not per-particle assignments.
- p ≈ 0.5 has three different causes, and only r separates them.

### Deliverable
Write `regfrac_results/REPORT.md` with these sections:
- a one-paragraph answer;
- the headline p with both error bars;
- a checks table for (a)–(i): value, threshold, pass/fail;
- the sensitivity numbers;
- which interpretation-table row applies, and why;
- the figures and the ChimeraX file;
- anything surprising.

Keep it factual: say what the data show and what they cannot show. Then give me a short summary in chat.
