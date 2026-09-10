# Public analysis workflow

This directory contains the portable publication-facing analytical workflow.

## Workflow order

1. `phase_v0_8_nutritional_metabolic_expansion.py`
2. `phase_v0_8_2_module_coherence_and_integration.py`
3. `phase_v0_8_3_final_recovery_axis_adjudication.py`
4. `phase_v0_8_4_3_final_visual_lock.py`

The intermediate v0.8.4 figure-development script is intentionally omitted because v0.8.4.3 is the final publication-facing visual workflow.

## Input boundary

Participant-level source data are not distributed in this repository. Users must obtain authorized proteomics and clinical inputs separately.

The v0.8.2.2 phase uses `resources/curated_legacy_module_mapping.tsv` as its participant-independent module-to-protein mapping resource.

Running the workflow with participant-level data can generate participant/sample-level intermediate files. These local outputs must not be committed to the public repository.

The earlier collaborator-coupled `jacobs_*` development scripts are intentionally excluded from the public package.
