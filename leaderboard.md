# Experiment leaderboard

Ranked by best held-out validation median haversine distance (km, lower is
better). All runs are fully deterministic (seed=42), 30-epoch runs unless
noted, on the same stratified train/val split (checked for leakage — see
below). Outputs live under `geo_guesser_runs/outputs/<name>/`.

| Rank | Experiment | Backbone | Params (% of 5M) | Key changes vs. baseline recipe | Epochs | Best val median (km) |
|---|---|---|---|---|---|---|
| 1 | **femto-160c-res512** | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | same as `femto-160c-lr5e5` but full native 512×512 input resolution instead of 256px crop | 30 | **103.5** |
| 2 | femto-160c-lr5e5 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | same as femto baseline but `lr=5e-5` (half) | 30 | 110.8 |
| 3 | femto-160c-wd3e3 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | same as femto baseline but `wd=3e-3` | 30 | 112.5 |
| 4 | femto-160c-wd3e4 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | same as femto baseline but `wd=3e-4` | 30 | 113.2 |
| 5 | femto-160c-cellheavy | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | femto baseline but `cell_weight=2.0`, `offset_weight=0.5` (shift emphasis toward classification) | 30 | 113.4 |
| 6 | femto-160c-lr5e5-wd3e3 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | stacked `lr=5e-5` + `wd=3e-3` — worse than `lr=5e-5` alone; LR/WD are entangled, didn't stack like `n_cells`+`wd` did | 30 | 113.4 |
| 7 | femto-160c-wd1e3-latfix | ConvNeXtV2-**femto** (pretrained) | 4,915,230 (98%) | same recipe as `cells-160c-wd1e3-latfix`, backbone swapped atto→femto | 30 | 113.8 |
| 8 | femto-160c-res320 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | `lr=5e-5`, input resolution 256→320 — worse; more pixels didn't help despite the error-analysis hypothesis | 30 | 113.9 |
| 9 | femto-192c | ConvNeXtV2-femto (pretrained) | 4,927,550 (99%) | femto baseline but 192 cells (vs 160) | 30 | 114.0 |
| 10 | femto-160c-aux10 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | femto baseline but `aux_weight=1.0` (country loss weight, vs default 0.3) | 30 | 114.1 |
| 11 | femto-noflip | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | femto baseline, `RandomHorizontalFlip` removed — worse, so flip stays | 30 | 116.9 |
| 12 | femto-128c | ConvNeXtV2-femto (pretrained) | 4,902,910 (98%) | femto baseline but 128 cells (vs 160) | 30 | 117.5 |
| 13 | femto-160c-aux06 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | femto baseline but `aux_weight=0.6` — worse; over-weighting country hurts cell classification | 30 | 118.8 |
| 14 | femto-160c-noblur | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | `lr=5e-5`, `GaussianBlur` removed from augmentation — much worse, blur is a real regularizer | 30 | 121.1 |
| 15 | cells-160c-wd1e3-latfix | ConvNeXtV2-atto (pretrained) | 3,443,254 (69%) | 160 cells, `wd=1e-3`, + latitude-corrected offset target (local east/north km instead of raw degrees) | 30 | 129.1 |
| 16 | cells-160c-wd1e3 | ConvNeXtV2-atto (pretrained) | 3,443,254 (69%) | 160 cells, flat `lr=1e-4`, `wd=1e-3` (10× default) | 30 | 131.2 |
| 17 | cells-160c | ConvNeXtV2-atto (pretrained) | 3,443,254 (69%) | 160-cell classification + offset regression (hard argmax), flat `lr=1e-4` | 30 | 133.1 |
| 18 | cells-128c-lion-lr1e5 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, Lion optimizer (`lr=1e-5`, `wd=1e-2`) instead of AdamW | 30 | 133.9 |
| 19 | cells-160c-cosine | ConvNeXtV2-atto (pretrained) | 3,443,254 (69%) | 160 cells, cosine LR decay `1e-4`→`1e-6` — worse than plain 160c, didn't stack | 30 | 134.4 |
| 20 | cells-128c-wd1e3 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, `wd=1e-3` (10× default) | 30 | 136.9 |
| 21 | cells-160c-lion-lr1e5 | ConvNeXtV2-atto (pretrained) | 3,443,254 (69%) | 160 cells, Lion (`lr=1e-5`, `wd=1e-2`) — worse than plain 160c or 128c-lion, didn't stack | 30 | 137.0 |
| 22 | cells-128c-cosine | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, cosine LR decay `1e-4`→`1e-6` | 30 | 137.7 |
| 23 | cells-256c | ConvNeXtV2-atto (pretrained) | 3,474,070 (69%) | 256 cells, flat `lr=1e-4` | 30 | 138.2 |
| 24 | cells-192c | ConvNeXtV2-atto (pretrained) | 3,453,526 (69%) | 192 cells, flat `lr=1e-4` | 30 | 138.9 |
| 25 | cells-128c-lr2e4 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, `lr=2e-4` | 30 | 141.9 |
| 26 | cells-128c-lion-lr3e5 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, Lion (`lr=3e-5`, `wd=1e-2`) | 30 | 142.1 |
| 27 | convnextv2-cells-vanilla-128c | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, flat `lr=1e-4`, AdamW default `wd=1e-4` (baseline for all `128c-*` rows above) | 30 | 142.5 |
| 28 | cells-128c-lr5e5 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, `lr=5e-5` | 30 | 142.9 |
| 29 | convnextv2-cells-country | ConvNeXtV2-atto (pretrained) | 3,422,710 (68%) | 96 cells fit *within each country* (8/country), classification gated by predicted country | 30 | 149.0 |
| 30 | convnextv2-cells-vanilla-96c | ConvNeXtV2-atto (pretrained) | 3,422,710 (68%) | 96 cells, flat (not country-gated) | 30 | 150.9 |
| 31 | cells-2level-20x8 | ConvNeXtV2-atto (pretrained) | 3,449,674 (69%) | two-level hierarchy: 20 coarse × 8 fine = 160 total cells (vs. one flat 160-way partition), `wd=1e-3` + latfix | 30 | 155.0 |
| 32 | cells-128c-lr3e4 | ConvNeXtV2-atto (pretrained) | 3,432,982 (69%) | 128 cells, `lr=3e-4` — too high, destabilized | 30 | 163.4 |
| 33 | convnextv2-cells-vanilla | ConvNeXtV2-atto (pretrained) | 3,407,302 (68%) | 48 cells | 30 | 180.2 |
| 34 | femto-160c-lr2e4 | ConvNeXtV2-femto (pretrained) | 4,915,230 (98%) | femto baseline but `lr=2e-4` — destabilized, unlike atto | 30 | 196.9 |
| 35 | convnextv2-cells-vanilla-24c | ConvNeXtV2-atto (pretrained) | 3,399,598 (68%) | 24 cells | 30 | 204.0 |
| 36 | convnextv2-lowlr-60ep | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | flat regression, flat `lr=1e-4` instead of `1e-3`, extended training | 60 | 246.3 |
| 37 | convnextv2-cosine-lowlr-60ep | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | flat regression, cosine LR decay `1e-4`→`1e-6`, extended training | 60 | 263.9 |
| 38 | convnextv2-cells-reweighted | ConvNeXtV2-atto (pretrained) | 3,407,302 (68%) | 48 cells, soft top-3 blend + end-to-end haversine loss instead of hard argmax | 30 | 264.2 |
| 39 | femto-plain-regression | ConvNeXtV2-femto (pretrained) | 4,853,630 (97%) | femto backbone, **plain flat-regression** (no cells) — isolates cells-vs-backbone contribution | 30 | 275.0 |
| 40 | gmm-convnextv2 | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | original `gmm` branch (flat regression + land/sea prior penalty), backbone swapped to ConvNeXtV2, flat `lr=1e-4` | 30 | 293.1 |
| 41 | convnextv2-lowlr | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | flat regression, flat `lr=1e-4` instead of `1e-3` | 30 | 296.6 |
| 42 | augmentation-cosine | MobileViT-x-small (pretrained) | 1,938,238 (39%) | flat regression, cosine LR decay `1e-3`→`1e-5` | 30 | 319.1 |
| 43 | efficientnet-lowlr-60ep | EfficientNet-B0 (pretrained) | 4,025,482 (80%) | flat regression, backbone swap, flat `lr=1e-4`, extended training | 60 | 319.2 |
| 44 | gmm | MobileViT-x-small (pretrained) | 1,938,238 (39%) | flat regression + land/sea GMM prior penalty | 30 | 395.2 |
| 45 | augmentation | MobileViT-x-small (pretrained) | 1,938,238 (39%) | flat regression, stratified split + stronger augmentation | 30 | 402.9 |
| 46 | gmm-head | MobileViT-x-small (pretrained) | 2,035,518 (41%) | flat regression + shared trunk head (Linear→BN→SiLU→Dropout) | 30 | 419.7 |
| 47 | cnn | SmallCNN (from scratch) | 3,624,238 (72%) | flat regression, MobileNetV2-style, trained from scratch | 30 | 683.5 |
| 48 | vit | SimpleViT (from scratch) | 3,260,334 (65%) | flat regression, plain ViT, trained from scratch | 30 | 1178.5 |
| 49 | convnextv2 | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | flat regression, backbone swap only, original flat `lr=1e-3` | 30 | 1182.8 |
| 50 | convnextv2-cosine | ConvNeXtV2-atto (pretrained) | 3,391,894 (68%) | flat regression, cosine LR decay `1e-3`→`1e-5` (starts too high) | 30 | 1184.3 |

## Key takeaways

- **We were backbone-capacity-limited after all — biggest single win since
  the original flat-regression→cells shift.** Every experiment so far used
  ConvNeXtV2-atto (3.4M params, 69% of the 5M budget) because it happened to
  win the very first backbone comparison, before the cells architecture even
  existed. Swapping to ConvNeXtV2-**femto** (4.92M, 98% of budget) with the
  exact same recipe (160 cells, `wd=1e-3`, latitude-corrected offset) took
  the best score from 129.1 to 113.8 km — an 11.8% relative drop, bigger
  than cell-count tuning, weight decay, or the offset fix individually.
  Checked several other backbones first (MobileViT-small, EfficientNet-B1/B2,
  RegNet, MobileNetV3-Large) — femto was the only one that both fits under
  5M with a 160-cell head *and* is meaningfully bigger than atto (pico is
  next up the ConvNeXtV2 family but already exceeds 5M alone).
- **Femto has different hyperparameter optima than atto — re-tuning on it
  found another win.** All earlier tuning (cell count, LR, wd, optimizer)
  was done on the smaller atto backbone. Re-swept on femto: it wants a
  **lower** LR (`5e-5` beats `1e-4`: 110.8 vs 113.8 km — new best) and is
  *more* sensitive on the high end (`lr=2e-4` destabilized it at 196.9 km,
  whereas that same LR was fine, even slightly better, for atto). `n_cells`
  stayed at 160 (128 and 192 both worse), and `wd=3e-3`/`wd=3e-4` each beat
  the `1e-3` baseline slightly (112.5 / 113.2), suggesting femto isn't very
  sensitive to weight decay in that range — unlike LR, where it clearly is.
  `lr=5e-5` and `wd=3e-3` haven't been combined yet.
- **Horizontal flip actually helps — settles a question parked since early
  in the project.** Removing it from femto's augmentation made things worse
  (116.9 vs 113.8 km), contradicting the original research notes' hypothesis
  that flipping destroys left/right-hand-traffic and signage cues. Keeping it.
- **Loss-weight reweighting, motivated directly by the country-confusion
  finding below, didn't help.** Raising `aux_weight` (country loss) to make
  the shared backbone learn more country-discriminative features was a
  reasonable hypothesis, but `aux_weight=0.6` made things clearly worse
  (118.8 km) and `aux_weight=1.0` was a wash (114.1 km, within noise of the
  113.8 baseline). Shifting emphasis from offset to cell classification
  (`cell_weight=2.0`, `offset_weight=0.5`) was also a wash (113.4 km).
  Suggests the country confusion documented below isn't a training-emphasis
  problem loss reweighting can fix — more likely a real visual-discriminability
  ceiling for a cluster of similar-looking countries at this model scale.
- **The backbone-capacity win is specific to the cells architecture, not a
  generic "bigger is better."** Ran femto with plain flat regression (no
  cells) under the same recipe: 275.0 km — better than atto's plain
  regression at the same 30 epochs (296.6 km) but nowhere near proportional
  to the 11.8% cells-architecture gain, and nowhere close to the 113.8 km
  cells result. More backbone capacity mainly pays off when there's a
  classification task (160-way) that can actually use the extra
  discriminative power — flat regression can't use it nearly as well.
  We're now within reach of the <100km target that prompted this whole
  line of investigation.
- **Latitude-corrected offsets gave another clean, small win.** The offset
  regression target was raw `(Δlat, Δlng)` degrees, but a degree of
  longitude isn't a constant distance — it shrinks by ~cos(latitude), a
  ~1.84x difference between Turkey (39N) and Iceland (65N) in this dataset.
  Reprojecting the target to local east/north kilometers before
  normalizing (instead of raw degrees) took the best config from 131.2 to
  **129.1 km**. Verified the projection is an exact round-trip before
  spending GPU time on it. Smaller win than cell-count or weight decay, but
  it's a targeted fix for a specific, identifiable flaw rather than a
  hyperparameter guess, so it stacked cleanly with no surprises.
- **The hierarchical target is the single biggest win so far.** Replacing
  flat lat/lng regression with cell classification + a small offset
  regression dropped the best score from 246.3 to 150.9 km — bigger than
  every LR/backbone tweak combined, in only 30 epochs. All three cell-count
  variants overfit within the 30-epoch budget (val loss climbing while
  train loss kept falling), so the saved checkpoint is always the best
  epoch, not the last one.
- **`n_cells` peaks at 160, then reverses.** Full sweep: 24 → 204.0, 48 →
  180.2, 96 → 150.9, 128 → 142.5, **160 → 133.1 km (best)**, 192 → 138.9,
  256 → 138.2. Past 160, fewer training images/cell (~55 at 192, ~41 at 256)
  starts hurting classification more than the finer geography helps.
- **Two-level hierarchy (coarse→fine) made things worse, not better** — 155.0
  km vs. 129.1 km for the flat 160-cell approach, at the same total cell
  budget (20 coarse × 8 fine = 160). The idea was sound in principle (let
  classification scale past 160 without any single softmax needing huge,
  thin classes), but in practice splitting into 20 coarse groups first and
  independently re-clustering each into 8 fine cells produced much less even
  cell sizes than one flat 160-way KMeans (min size 4 vs. 20) — some coarse
  groups get force-split into fine cells regardless of how few points they
  have. It made exactly the problem it was meant to solve worse. Might be
  salvageable with a different coarse/fine split (fewer, bigger coarse
  groups) but not revisited given the backbone swap below was a much bigger
  and cleaner win.
- **Weight decay stacks with more cells; LR schedule and Lion don't.** Tried
  combining each 128-cell win with `n_cells=160`: `wd=1e-3` stacked cleanly
  (133.1 → **131.2 km**, new best) — regularization and geographic
  resolution are addressing different problems, so they compound. But
  cosine (134.4 km) and Lion `lr=1e-5` (137.0 km) both got *worse* at 160
  cells than they were at 128, and worse than plain 160c flat AdamW. The
  "right" LR trajectory is apparently coupled to how many classes you're
  training on — a fix tuned for one `n_cells` doesn't necessarily transfer
  to another, unlike weight decay which is more of a general-purpose
  regularizer.
- **LR sweep confirms a real ceiling**: `lr=2e-4` edges out `1e-4` slightly
  (141.9 vs 142.5 at 128 cells) but `lr=3e-4` destabilizes training (163.4 km).
- **Lion optimizer is a legitimate alternative to AdamW at the right scale**:
  `lr=1e-5, wd=1e-2` reached 133.9 km at 128 cells, almost tying AdamW's
  best — but it didn't generalize to 160 cells (see above), and `lr=3e-5`
  was mediocre (142.1 km), so it's just as LR-sensitive as AdamW, at a
  different scale.
- **Country-gating cells gives a small real improvement, but less than just
  adding more cells.** Built cells *within* each country (8/country = 96
  total, gated by predicted country at inference, loss masked by true
  country during training) — 149.0 km vs. 150.9 km for the flat 96-cell
  version, a genuine but modest ~1.3% gain. It still loses to simply using
  128 flat cells (142.5 km) at a similar total budget. Suggests wrong-country
  mistakes weren't a major error source to begin with (country classification
  was probably already fairly reliable), so the country boundary constraint
  recovers only a small slice of what more geographic resolution gets you
  directly.
- **Reviving the GMM land-prior idea on ConvNeXtV2 didn't help either**
  (`gmm-convnextv2`, 293.1 km) — confirms this is fundamentally a
  flat-regression technique (land-prior penalizes implausible predictions,
  it doesn't change what's being predicted) and flat regression plateaus far
  above the cells approach regardless of backbone or regularization. Closing
  this avenue.
- **Soft top-k reweighting made it worse, not better** (264.2 km vs. 180.2).
  The hypothesis was that hard `argmax` cell selection risks blowing up the
  error on a wrong cell pick, and blending the top-3 cells' medoids would
  fix that. Instead it hurt — most likely because this dataset's cells are
  geographically well-separated (roughly country-sized clusters with strong
  visual cues), so wrong-cell picks were probably already rare under hard
  argmax. Blending pulled *every* prediction — including the already-correct
  confident majority — toward a weighted average of neighboring cells,
  trading a rare catastrophic error for a small systematic error on most
  predictions. The training signal also changed: vanilla gives the offset
  head a clean, independent regression target (relative to the true cell);
  reweighted couples the offset and cell-classification heads together
  through the blended sum, which may just be a harder optimization problem.
  Sticking with hard argmax.
- **Capacity isn't the bottleneck.** The best flat-regression model uses 68%
  of the 5M-param budget; MobileViT variants at 39% aren't far behind. Every
  from-scratch or naively-swapped bigger backbone did worse.
- **Pretrained backbones need a low starting LR, not just a decaying one.**
  `convnextv2` (flat `1e-3`) and `convnextv2-cosine` (`1e-3` decaying slowly)
  both landed around 1180 km — nearly identical, because cosine annealing
  doesn't drop below ~`1e-4` until epoch ~26, by which point the pretrained
  features are already wrecked. A flat low LR from step one (`convnextv2-lowlr`)
  fixed it immediately: 1182.8 → 296.6 km.
- **More epochs helped once the LR was fixed** — `convnextv2-lowlr` was still
  improving at epoch 30, and running it to 60 epochs took it to 246.3 km, the
  current best. It may not have fully converged even then.
- **Decaying an already-good LR further hurt, not helped.** `convnextv2-cosine-lowlr-60ep`
  (`1e-4`→`1e-6`) landed at 263.9 km, worse than the flat `1e-4` run (246.3 km).
  Its train_km kept dropping to 77 by epoch 60 while val_km plateaued at
  ~265-269 from epoch 51 on — the decay let it overfit further in the same
  epoch budget without any matching val improvement. Once the LR is already
  stable and non-destructive, tapering it further isn't free.
- **EfficientNet-B0 didn't beat ConvNeXtV2-atto**, even with the identical
  proven recipe (flat `lr=1e-4`, 60 epochs): 319.2 km vs 246.3 km. Its curve
  was still improving at epoch 60 with a larger train/val gap than ConvNeXtV2
  shows at the same point, so it may not have converged as cleanly — not a
  slam-dunk backbone swap here regardless.
- **Adding head capacity didn't help** (`gmm-head`, worse than plain `gmm`) —
  under the current flat-LR recipe it just overfit faster. Worth retrying
  once paired with a fixed LR schedule, not on its own.
- **Leakage check: clean.** 11,758 training images — 0 duplicate filenames,
  0 duplicate `(lat,lng)` pairs, 0 byte-identical duplicate files. No exact
  duplicate leakage across the train/val split (near-duplicate photos of the
  same spot from a different angle aren't ruled out, only exact duplicates).
## Error analysis (on `femto-160c-wd1e3-latfix`, the then-current best)

Scored the val set directly (1175 images, ground truth known) rather than
just looking at the aggregate median. Median (113.8 km) and mean (473.2 km)
diverge hugely — the error distribution is sharply bimodal, driven almost
entirely by one factor:

| | Count | Median error | Mean error |
|---|---|---|---|
| Predicted cell in the **right** country | 854 (72.7%) | 57.1 km | 147.3 km |
| Predicted cell in the **wrong** country | 321 (27.3%) | 1143.2 km | 1340.5 km |

Country confusion isn't random — it's concentrated in one cluster of
visually-similar, contiguous Central/Western European countries (Germany↔
Poland, Poland↔France, UK↔France, Italy↔Turkey, France↔Germany/UK/Spain),
while geographically/visually distinctive countries (Iceland 90% <200km,
Belarus 80%, Norway/Finland 70%) are rarely confused with anything. Germany
is the single worst country (448.8 km median, only 30% <200km).

Cell-classification confidence is a strong, well-calibrated reliability
signal (mean error drops monotonically from 973.5 km at confidence <0.3 to
103.2 km at confidence >0.9) — but cell training-set size has a much weaker
relationship with error, meaning `n_cells` tuning was already near the right
tradeoff; **country-level confusability, not cell sparsity, is the dominant
remaining problem.**

**Tried a confidence-gated fallback based on this** (below threshold, fall
back to the predicted country's centroid instead of the cell+offset) —
tested directly against the trained checkpoint, no retraining needed. It
made things *worse* at every threshold tried, including the lowest-confidence
bucket. Two reasons: a country centroid is a single fixed point that ignores
the image entirely, so even a "confused" cell prediction (which still uses
the actual visual content) tends to beat it; and `country_head`'s reliability
turned out to be correlated with `cell_head`'s, not independent of it, so
the fallback isn't drawing on a genuinely more-trustworthy signal. Good
result to have tested empirically rather than assumed.

## Final round: three ideas, two negative — then resolution flipped the story

Tried to close the remaining gap to <100km with three targeted follow-ups
on top of the `femto-160c-lr5e5` baseline (110.8 km):

- **`lr=5e-5` + `wd=3e-3` stacked**: 113.4 km, *worse* than `lr=5e-5` alone.
  Unlike `n_cells`+`wd` (which stacked cleanly earlier because they address
  different problems), LR and weight decay are more entangled — both affect
  effective step size, so pushing both away from their individually-tuned
  values overshot.
- **Removing `GaussianBlur`**: 121.1 km, substantially worse (~9.4%
  relative). The opposite of the reasoning that blur was destroying useful
  detail — it's acting as a real regularizer against overfitting to noisy
  fine detail, not an obstacle.
- **Input resolution 256→320**: 113.9 km, worse — seemed to close the
  resolution question with a "no." **But then tried the full native
  512×512 resolution (no downsampling at all) and got 103.5 km — a real,
  substantial win (6.6% relative), the new best.** The resolution story
  isn't "more pixels helps discriminate countries" in the abstract (which
  320px already refuted) — it's about *avoiding the downsampling loss*
  from the native 512×512 source. At 256px every training image loses
  ~75% of its pixels to downsampling; at 320px it's still losing most of
  it; at 512px, crops near full-image are preserved with zero resizing.
  320 was a partial step that wasn't enough to matter either way; going
  all the way to native resolution is what actually paid off. The blur
  finding above doesn't contradict this — blur is a targeted, controlled
  perturbation used as a regularizer, not the same thing as blanket
  information loss from downsampling every training image by 4x.

`femto-160c-res512`'s training curve shows the same overfitting signature
seen throughout the project (best epoch 24 at 103.5 km, val_km bouncing
103–122 afterward while train_km keeps falling) — the same pattern that
`wd=1e-3→3e-3` fixed before, but never tested paired with 512px. Two
follow-ups in progress: `res512 + wd=3e-3` and `res512 + lr=3e-5`.

**Current best: `femto-160c-res512`** — ConvNeXtV2-femto, 160 cells,
`lr=5e-5`, `wd=1e-3`, latitude-corrected offset target, full native
512×512 input resolution, **103.5 km**. Two follow-ups in progress
(`wd=3e-3`, `lr=3e-5`, both at 512px) before finalizing; see
`writeup/report_draft.md` for the full writeup.
