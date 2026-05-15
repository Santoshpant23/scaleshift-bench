# Email to Dr. Ruben Fernandez-Beltran (Universitat Jaume I, RicePAL corresponding author)

**To:** rufernan@uji.es
**Cc:** (none)
**Subject:** RicePAL Nepal Terai data access for foundation-model evaluation

---

Dear Dr. Fernandez-Beltran,

I am Santosh Pant, a rising senior in Computer Science and Data Science at Knox College (Galesburg, IL), and a Knox Summer Scholars fellow this year. I am building an evaluation benchmark for the latest geospatial foundation models (Clay v1.5, Prithvi-EO-2.0, AnySat, TerraMind) on smallholder-agriculture tasks in Nepal's Terai, where my family farms, with the goal of submitting to NeurIPS Datasets & Benchmarks 2027.

I came across your 2021 *Remote Sensing* paper on RicePAL (doi:10.3390/rs13071391) while searching for Nepal-Terai-specific rice labels. The 2016–2018 multi-year coverage and the alignment with Sentinel-2 are exactly what I need to construct a size-stratified evaluation of how foundation models handle the smallholder rice fields that dominate the Terai.

Would you be willing to share the underlying rice/non-rice mask data for academic use? I would happily cite RicePAL as the source in any resulting paper. Specifically I am hoping to:

1. Extract field-level polygons from the masks (watershed + morphological cleanup, possibly SAM2 refinement).
2. Stratify by field size in bins of <0.1, 0.1–0.3, 0.3–0.5, 0.5–1, and >1 hectares.
3. Evaluate the four foundation models zero-shot on rice-vs-non-rice classification across those strata, then explore a scale-aware adaptation method to close the gap on the smallest fields.

I am happy to share the project code (github.com/Santoshpant23/scaleshift-bench) and the preliminary results from my public-data baseline (ESA WorldCover cropland on Chitwan, Dhanusha, Bardiya) as soon as they are ready.

If access is gated or requires a research agreement, please let me know what process you would suggest. If there is more recent RicePAL data covering 2019–2024 that you would consider sharing for the same purpose, that would also be of strong interest.

Thank you very much for your time and for putting together a dataset that fills a real gap for Nepal.

Best regards,
Santosh Pant
Class of 2027, Knox College
santoshpant613@gmail.com
GitHub: github.com/Santoshpant23/scaleshift-bench

---

## When to send

Same trigger as the Panday email: after preliminary cross-FM zero-shot results on the public-data baseline. The credibility delta between "I want to do X" and "I built X on public data, here are the limits, your data unlocks Y" is large.

## What a positive response looks like

- Direct file/link share → integrate as `data/labels/ricepal/` and re-extract polygons over Terai.
- "Use this Zenodo/Mendeley link." → same.
- "Email a colleague who manages the data." → forward promptly.

## What a no/non-response looks like

- Two weeks of silence → one-paragraph follow-up.
- "Sorry, data is unavailable." → drop, stick with WorldCover + JECAM (if Panday responds positively) for the published benchmark; note the absence in the limitations section.
