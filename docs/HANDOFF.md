# CausalRivers project: handoff (end of session 1)

## Goal
Causal discovery on CausalRivers (Stein et al., ICLR 2025): classical baselines, then new first-principles methods, then an arXiv paper and hands-on causal inference course notes. Work runs in Colab/Kaggle, pushed to GitHub (Mathindata).

## Files
- `01_eda_causal_lens.ipynb`: done, executed. EDA from a causal perspective.
- `02a_baselines_executed.ipynb`: 26 baselines in 9 classes, math before each method, run on 10 of 12 sets (30 samples each), findings written.
- `02a_baselines_executed.ipynb` was later re-run in Colab on all 12 sets (the uploaded version is the reference).
- `02b_shortcut_test.ipynb`: done. Magnitude-shortcut test, Random fix, bootstrap confidence intervals. Builder: `build_nb02b.py`.
- `02_parts_n30.zip`: cached predictions for the 10 finished sets. Unzip into `results/`.
- `m02_methods.py`, `build_nb02.py`: method library and notebook builder.

## Settings
East Germany, 6h means, linear interpolation, max lag K = 3, 30 samples per set (SEED = 42). Headline metric: mean per-sample AUROC (checked equal to the official scorer). Also skeleton AUROC (detection) and orientation accuracy.
Env vars: N_PER_SET, N_JOBS, SETS_ONLY (comma list of sets).

## Key findings so far
From notebook 01:
- The ground truth is almost an in-forest: about 90% of stations have exactly one downstream neighbour.
- Rain acts as a hidden confounder. Co-parents and 2-hop pairs overlap heavily with true edges in dependence.
- At daily resolution, 74% of true edges peak at lag 0.
- 91% of 15-min changes are exactly zero, and excess kurtosis is in the hundreds.
- Shortcuts: "bigger river is downstream" orients 89-95% of true edges, elevation 95-99.8%, versus 60% for lagged cross-correlation.
- The metadata README swaps X and Y: X is latitude.

From notebook 02a:
- Elevation (metadata) has the best mean AUROC (0.80). The best data-driven method is PCMCI with a rank transform (0.73).
- Detection is the bottleneck: the best skeleton AUROC is 0.73.
- The released "confounder" sets keep the split parent (an observed confounder), unlike the paper's text. We added `confounder_hidden` sets.
- The paper's 1 - p score for PCMCI (0.67) is below the effect-size score (0.69).
- RESOLVED in 02b. The paper's VAR/VARLiNGAM scores come from raw (unnormalized) data: best run unnormalized on 91% / 100% of sets in their own grid.
  - Raw VAR: AUROC 0.77, orientation 0.84, skeleton 0.57, agrees with the RP size rule on 87% of pairs.
  - z-scored VAR: 0.59 / 0.52 / 0.65.
  - Granger F is identical under all scalings (control).
  - Conclusion: raw-scale coefficient methods are physics-informed, not data-only.
- Bootstrap CIs (30 samples/set) are about ±0.025, so the data-only top group is tied at about 0.71 to 0.73: PCMCI rank, Granger, CC.
- Two tracks from now on: data-only (best about 0.72) and physics-informed (elevation 0.80, raw VAR 0.77).

## Next steps
1. (done) Random fix, VAR gap, CIs.
2. Colab: mount Google Drive as WORK so data, cache and results persist.
3. Notebook 03: tune on Bavaria (the official train graph), report on East Germany. Include a resolution sweep and normalization on/off.
4. Phase 2 method ideas:
   - tree / arborescence prior (Chu-Liu/Edmonds on pairwise scores)
   - rank-based conditional tests
   - event-based lag consistency
   - mass-balance routing model
   - a latent rain driver
5. Deep baselines (CDMI, CP) on GPU later. Then the paper and course notes; notebooks 01 and 02 already map to lessons.
