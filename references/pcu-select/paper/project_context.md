# Project Context: PCU-Select

## Identity

PCU-Select amortizes PEFT-conditioned data selection across a registry by learning reusable sample, task, and structured configuration representations while preserving the quality of per-PEFT recomputation.

## Venue

- **Target:** AAAI 2027
- **Main-text limit:** 7 pages, with references beginning on page 8
- **Format:** Two-column AAAI machine-learning paper; anonymous submission

## Contributions and Evidence

1. **PEFT-conditioned utility.** Data-value rankings and selected-subset quality depend on the trainable PEFT subspace.
   - Evidence: Figure 1 and Supplementary Table S10.
2. **Intervention-site amortization.** A shared 24-site coordinate system and structured PEFT code retain target specificity without rebuilding a gradient datastore.
   - Evidence: Method, Figure 2, and Supplementary Table S9.
3. **Registry-scale quality and cost.** PCU-Select near-ties per-PEFT LESS and reduces marginal onboarding cost for a new four-task PEFT from 63.2 to 1.51 GPU-hours.
   - Evidence: Tables 1 and 2.

## Locked Decisions

1. Preserve the existing AAAI section structure and anonymous-submission settings.
2. Do not introduce new results, citations, datasets, baselines, or implementation details without source evidence.
3. Keep the complete main text within pages 1--7 and begin References on page 8.
4. Treat Prefix/P-Tuning as outside the supported zero-shot regime. Report
   calibrated results only with externally supplied compatible labels and retain
   the residual gap as an applicability boundary.

## Open Questions

1. Whether the evidence generalizes beyond the evaluated backbone family and four tasks.
2. Whether the native backend can generate Prefix/P-Tuning calibration labels;
   current calibrated results rely on externally supplied labels.
