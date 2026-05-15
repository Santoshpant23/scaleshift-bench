# Email to Dr. Uma Shankar Panday (Kathmandu University, JECAM Nepal site PI)

**To:** uspanday@ku.edu.np
**Cc:** (none yet — once you have a faculty advisor for ScaleShift, cc them)
**Subject:** Question about JECAM Nepal-Dhanusha phenology records

---

Dear Dr. Panday,

I am Santosh Pant, a rising senior in Computer Science and Data Science at Knox College (Galesburg, IL). I grew up in Mahendranagar, Kanchanpur, in the far-western Terai, where my parents farm. This summer I am a Knox Summer Scholars fellow benchmarking the latest geospatial foundation models (Clay, Prithvi-EO-2.0, AnySat, TerraMind) on smallholder-agriculture tasks in the Nepal Terai, with the goal of submitting to NeurIPS Datasets & Benchmarks 2027.

I have been reading about the JECAM Nepal-Dhanusha/Mahottari site that you lead, and I would like to ask one question that would help me scope the project.

The 2020–2022 ODK in-situ crop labels your team collected (rice, sugarcane, maize, wheat, lentils) — do they include phenological-stage annotations alongside the crop-type labels? By "stage" I mean any record of emergence / vegetative / heading-reproductive / maturity, BBCH numbers, or even visual stage notes that the enumerators logged. The records do not need to be published or polished — even an indication that "yes, that information was collected but lives in an unpublished CSV" would tell me whether to scope a temporal-generalization benchmark over Nepal Terai or stay with size-stratified cropland classification.

For context: I built a benchmark pipeline this week that pulls Sentinel-2 + Sentinel-1 + ESA WorldCover labels over three pilot Terai districts (Chitwan, Dhanusha, Bardiya) and runs zero-shot evaluation across the four foundation models stratified by field size. I would be glad to share the preliminary results when they are ready, and I would cite any JECAM data appropriately if you are willing to share.

I understand you receive many such requests. Even a one-line reply ("yes stages exist" / "no stages, crop-type only") would be enormously helpful. If there is a better person on your team to direct this to, I will happily reach out to them.

Thank you for your time, and for the work your group has done in the Madhesh Terai.

Best regards,
Santosh Pant
Class of 2027, Knox College
santoshpant613@gmail.com
GitHub: github.com/Santoshpant23/scaleshift-bench

---

## When to send

After the preliminary cross-FM zero-shot results are in (≈ end of week 1 of Phase 1). Send with a one-page PDF of the F1-vs-field-size plot attached if available. Without preliminary results the email is weaker; with them, the credibility is set.

## What a positive response looks like

- "Yes, stages were recorded. Here are the relevant CSV files." → pivot to PhenoShift-Bench planning, set up a follow-up call.
- "Crop-type only, no stages." → JECAM polygons still useful for Sub-Hectare; ask separately about polygon access.
- "Some 2022 records have stages, earlier years don't." → partial-data scenario, decide whether 1 year is enough for PhenoShift.

## What a no/non-response looks like

- No reply within 2 weeks → send a one-paragraph follow-up referencing the original. After that, drop and continue with Sub-Hectare on public data.
- "Data is restricted." → ask whether a research agreement is possible; if not, drop.
