# Crawford / KGS Eastern Kentucky Supervised-Baseline Briefing

Prepared for the CCNY Media Lab LSM meeting (eastern Kentucky pivot). Scope: what
Matthew M. Crawford and the Kentucky Geological Survey (KGS) have established as
the supervised landslide-susceptibility baseline for our 8-county eastern KY study
area. Every uncertain claim is flagged **[UNCERTAIN]**.

---

## 1. TL;DR — the established eastern-KY supervised baseline

Crawford and the KGS have, over a sequence of papers (2021 → 2022 → 2025),
established a **statistics/classical-ML supervised baseline** for eastern Kentucky
landslide susceptibility built on **lidar-derived geomorphic variables** (slope,
curvature, plan curvature, terrain roughness, aspect) fed to a **two-step model: a
bagged/decision-tree classifier followed by logistic / multinomial logistic
regression**, trained on **KGS point/polygon landslide inventories** and evaluated
by **ROC-AUC**. The headline performance band to know is **ROC-AUC ≈ 0.78–0.83**:
the original Magoffin-County model (2021) reported **AUC = 0.83**, and the most
recent iterative study (Crawford, Koch & Dortch 2025, *Natural Hazards*) reports a
working range whose **lowest, best-mapping configuration scored AUC = 0.78**. This
is the number our SSL deep-learning approach must be measured against. Their
explicit framing is "**limited resource / regional scale**" — i.e. they optimize
for practicality with classical statistics, not deep learning, which is precisely
the methodological gap our SSL + label-scarcity work targets.

---

## 2. The 2022 paper (read in full, local PDF)

**Citation (verified from PDF):** Crawford, M.M.; Dortch, J.M.; Koch, H.J.; Zhu, Y.;
Haneberg, W.C.; Wang, Z.; Bryson, L.S. "Landslide Risk Assessment in Eastern
Kentucky, USA: Developing a Regional Scale, Limited Resource Approach." *Remote
Sensing* **2022**, *14*(24), 6246. https://doi.org/10.3390/rs14246246
(Received 1 Nov 2022; published 9 Dec 2022.)

**Important scoping note:** This 2022 paper is a **risk-assessment** paper, not a
new susceptibility-model paper. Its susceptibility (hazard) layer is **imported
from the earlier Crawford et al. 2021 model** (ref [43] in the PDF). The headline
**AUC = 0.83 belongs to that 2021 model**, quoted inside the 2022 paper. The 2022
paper's own contribution is the **Risk = Hazard × Vulnerability × Consequence**
socioeconomic mapping, which has no AUC of its own.

### Study area & extent
- **Five eastern KY counties: Magoffin, Johnson, Floyd, Martin, Pike** — the Big
  Sandy Area Development District. (Note: these only **partially overlap** our
  8-county set; our set is Wolfe, Lee, Owsley, Breathitt, Knott, Perry, Leslie,
  Letcher — overlap is essentially just the broader Appalachian Plateau region,
  not the same counties.)
- Extent **5,136 km²**, population 140,215. Appalachian Plateau; **mean slope
  angle 24°–25°** (cf. our stated study-area mean ~21°).
- Bedrock: flat-lying sandstone, shale, siltstone, **coal**, underclay; colluvium
  0.5–5 m thick; landslides mostly thin (<3 m) translational slides and rotational
  slumps that can mobilize into debris flows.

### Landslide inventory
- KGS landslide inventory database (Crawford 2022; uknowledge.uky.edu/kgs_data/7).
- The **susceptibility model's training inventory** (from the 2021 paper) was
  **N > 1,054 landslides in Magoffin County**, mean landslide area **6,400 m²**,
  with an **equal number of landslide (1) and non-landslide (0)** samples — a
  **balanced binary** training table. Slide type/age were not determined.

### Conditioning factors / features (susceptibility model)
- **Geomorphic variables from a 1.5 m lidar-derived DEM: slope, curvature, plan
  curvature, terrain roughness, aspect.** "**Eight variables were significant
  (p < 0.05)**" in the logistic regression. (The 2022 paper lists five named
  variables; the "eight significant" count comes from the 2021 model.)

### Model(s)
- **Two-method combination:** a **bagged-tree model** (predicts a weighted
  classification, ranks variable importance) **+ logistic regression** (estimates
  per-cell probability of landslide occurrence). Output is a continuous
  probability map (0–1).

### DEM resolution
- **Susceptibility (primary) approach: 1.5 m airborne lidar DEM** (susceptibility
  output then handled at 3 m pixels).
- **"Limited resource" comparison approach: 30 m global SRTM DEM**, slope-only
  hazard input (resampled to 3 m). This coarse approach was shown to be markedly
  **worse / less consistent** than the lidar-susceptibility approach.

### Validation & headline numbers (quote exactly)
- **ROC-AUC = 0.83** for the lidar-based logistic-regression susceptibility model
  (the only model-performance metric in the paper; inherited from the 2021 model).
- Risk-map outcome: **64.1% of the study area classified as moderate-to-high
  socioeconomic risk** (susceptibility-based map). Risk in the low class dropped
  ~39% on average when switching to the coarse 30 m slope-based hazard input.
- Validation scheme for the AUC is **not detailed in this paper** (it points to the
  2021 source paper). **[UNCERTAIN]** whether spatial CV was used.

### "Limited resource" data-scarcity handling
- Core thesis: rank approaches along a **quantitative↔qualitative continuum** and
  pick the most useful combination realistic to produce. They deliberately
  contrast a **data-rich lidar susceptibility hazard** vs. a **data-poor 30 m
  slope-only hazard**, plus simplified exposure/vulnerability (vulnerability fixed
  at worst-case **V = 1**). Conclusion: minor improvements in input data quality
  (lidar vs SRTM; better exposure) produce large gains in map usefulness.

### Stated limitations (esp. our confounders)
- **Coal mines:** development on **reclaimed coal-mine sites** is explicitly noted
  as part of the terrain; the paper flags **over-prediction in valley bottoms /
  engineered embankments** (flat, dense areas near toe-slopes) — the same class of
  anthropogenic-terrain false-positive problem we worry about.
- **Bedrock cliffs:** not called out by name in this 2022 paper. **[UNCERTAIN]**
- Other limits: no runout/landslide-behavior data; vulnerability assumed = 1;
  exposure (powerlines, water/sewer) incomplete; population timing static; no
  rainfall-threshold/time-dependent triggering.

---

## 3. The "Crawford et al. (2025)" paper — IDENTIFIED (high confidence)

The 2025 paper Te Pei flagged is, with high confidence:

**Citation (verified via Springer/Natural Hazards):** Crawford, M.M.; Koch, H.J.;
Dortch, J.M. "Evaluating map quality and model performance through iterative
statistics-based landslide susceptibility in eastern KY." *Natural Hazards*
**2025**, *121*, 11633–11661. https://doi.org/10.1007/s11069-025-07255-7
(Published May 2025.)

What is **verified** from the abstract/landing page:
- **Method:** the same **two-step ML pipeline — a decision-tree algorithm followed
  by multinomial logistic regression** — applied **iteratively** to a new eastern
  KY area. This is a direct lineage from the 2021/2022 model.
- **Goal:** test **geomorphic variables not previously used** for this region, and
  examine **how the landslide inventory is sampled/used** (geographic distribution;
  balanced vs. unbalanced landslide/non-landslide tables).
- **Headline result (quote exactly):** adding geomorphic variables and changing
  inventory utilization **significantly affects both ROC-AUC and map quality**. A
  **geographically distributed inventory + an unbalanced binary table produced the
  LOWEST ROC-AUC (0.78) but the HIGHEST-quality map.** Key qualitative finding:
  **"more variables do not necessarily improve model performance or map quality,"**
  and tuning the **landslide/non-landslide balance by adding more non-landslides
  accounts for uncommon landscape features and anthropogenic alterations** (this is
  effectively their coal-mine / cliff / mine-spoil false-positive mitigation).

What is **uncertain / not visible behind the paywall** — flag at the meeting:
- **[UNCERTAIN]** Exact county list of the 2025 study area (abstract says "eastern
  KY"; **[UNCERTAIN]** whether it is our 8 counties or the July-2022 footprint).
- **[UNCERTAIN]** Whether the inventory is specifically the **July 2022 storm
  inventory** (1,000+ landslides). The separately confirmed July-2022 inventory
  (KGS, >1,000 landslides across Clay, Leslie, Perry, Breathitt, Knott, Letcher;
  documented by field recon + NDVI) is a **related but distinct** product; the 2025
  paper points to the general KGS inventory (uknowledge.uky.edu/kgs_data/7).
- **[UNCERTAIN]** Full conditioning-variable list, exact DEM resolution (likely the
  same lidar lineage, but not confirmed in the excerpt), validation scheme, and the
  **upper end of the AUC range** (only the lowest-AUC = 0.78 config is quoted in
  the abstract; the higher-AUC configs scored above 0.78 but the exact ceiling is
  not visible — **[UNCERTAIN]**).

**Bottom line on identification:** This is almost certainly the intended "Crawford
et al. 2025" baseline (right author, right region, right method lineage, 2025,
classical supervised ML). If the meeting needs the buried specifics (counties, DEM,
full variable list, AUC ceiling), someone should pull the full PDF — the
abstract-level facts above are solid, the per-table numbers are not yet in hand.

**Closest alternative candidates** (in case Te Pei meant a different "2025" paper):
- Crawford et al. (2023), "Reconnaissance of landslides and debris flows associated
  with the July 2022 flooding in eastern Kentucky" — the KGS **inventory** report
  (>1,000 landslides; field + NDVI). This is the **data** paper, not a model paper.
- A 2025/2026 arXiv preprint on **AlphaEarth satellite embeddings for LSM with deep
  learning** appeared in search results but is **not Crawford/KGS** and **[UNCERTAIN]**
  if related — likely irrelevant.

---

## 4. How this positions our SSL work

- **AUC target to beat: ≈ 0.78–0.83.** The defensible single number to quote as
  "the eastern-KY supervised baseline" is **ROC-AUC ~0.78–0.83** (0.83 from the
  2021/2022 lidar model; 0.78 as the 2025 best-map configuration). Per our own
  multi-seed rule, we should compare our SSL+ResNet-18 against this band using
  **mean ± SD over ≥3 seeds**, never a single run — and a win means clearing the
  upper edge of this band by more than our seed-to-seed SD.
- **Match their features, then differentiate.** Their inputs are **lidar geomorphic
  derivatives: slope, curvature, plan/profile curvature, terrain roughness,
  aspect** (+ topographic indices like TWI). Our current 13-channel terrain stack
  already overlaps heavily (aspect, plan/profile curv, slope, twi, spi,
  elevation…). For a fair head-to-head we should **align on lidar-derived geomorphic
  channels at matched resolution** (note: they use 1.5 m lidar; our pipeline is
  pivoting to Qianyi's **10 m** features — flag this resolution gap, it is a real
  confound, do **not** claim parity across resolutions).
- **Their model is classical (trees + logistic regression), ours is deep SSL.**
  That is the headline contrast: they explicitly brand their work "**limited
  resource**" and use statistics precisely because it is robust with small,
  practitioner-grade pipelines. **SSL pretraining on unlabeled terrain is the
  deep-learning answer to the same label-scarcity problem** — this is our
  differentiation, not a contradiction of their framing.
- **Coal mines & anthropogenic terrain are the shared confounder — and they have a
  baseline mitigation we should benchmark against.** The 2025 paper's key trick is
  **adding more non-landslide samples (unbalanced table) to teach the model about
  "uncommon landscape features and anthropogenic alterations"** (surface mines,
  spoil, embankments). Our equivalent levers are **PU-bagging negative selection**
  and **spatially-blocked CV**; we should explicitly show our pipeline handles
  surface-mine / bedrock-cliff false-highs at least as well.
- **Bedrock cliffs & valley-bottom over-prediction** are documented failure modes
  (2022 paper flags engineered-embankment / toe-slope over-prediction). Our
  **boundary-aware valid-context mask channel** and spatial blocking are the
  features to point to as our handling of edge/anthropogenic artifacts.
- **The label-scarcity + point-label gap is exactly our niche.** Their inventories
  are point/polygon and they note inventory sampling dominates results. Eastern KY
  gives 1,000+ **point** labels needing a **point-to-patch** strategy — a regime
  where (a) the supervised baseline is non-trivial (~0.78) but (b) labels are
  scarce and noisy, which is the **only condition under which SSL is justified**
  (per CLAUDE.md §1). This is a much stronger proving ground for SSL than the small,
  urban NYC set.

---

### Source URLs
- 2022 paper (local PDF, verified): https://doi.org/10.3390/rs14246246
- 2025 paper (Natural Hazards): https://link.springer.com/article/10.1007/s11069-025-07255-7
- KGS July 2022 inventory report news: https://www.uky.edu/KGS/news/2023_July2022-Landslides.php
- KGS landslide research hub: https://kygs.uky.edu/research/landslides/
- 2021 source model: Crawford et al., *Q. J. Eng. Geol. Hydrogeol.* 2021, 54 (ref [43] in the 2022 PDF)
