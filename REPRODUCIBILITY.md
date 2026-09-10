# Reproducibility

## Public release layer

This repository preserves publication-facing aggregate outputs and a portable analytical workflow for a longitudinal secondary analysis of serum proteomics during tuberculosis treatment.

The release includes:

1. sanitized aggregate module-level results;
2. recovery-axis feature membership and loadings;
3. aggregate within-group recovery estimates;
4. BMI-by-time interaction-model summaries;
5. recovery-axis validation statistics;
6. the locked publication figure;
7. generic input-data templates;
8. a participant-independent curated module-to-protein mapping; and
9. four portable scripts representing the final analytical path.

## Analytical workflow

The public workflow is:

`v0.8 -> v0.8.2.2 -> v0.8.3 -> v0.8.4.3`

The scripts are stored in `scripts/`.

The v0.8.2.2 phase uses `resources/curated_legacy_module_mapping.tsv`.

The intermediate v0.8.4 figure-development script is intentionally excluded because v0.8.4.3 represents the final publication-facing visual workflow.

## Participant-level inputs

Participant-level processed proteomics and clinical data are deliberately excluded from this repository.

The corresponding source-study mass-spectrometry data are associated with:

- ProteomeXchange/PRIDE: PXD040546
- MassIVE: MSV000091392

The collaborator-supplied processed abundance matrix, clinical metadata, TMT sample-linkage files, and reconstructed participant-level analytical datasets are not redistributed.

Accordingly, the numerical results cannot be regenerated solely from the public files included here. Re-execution requires appropriately obtained source inputs.

## Public code derivation

The public scripts were derived from locked private-project Git commit:

`3628c9e76dae153200c244a67c15386cb81e18b4`

Only the portability and public-interface changes required for release were introduced:

- fixed local project-root paths were replaced by repository-relative root inference;
- the v0.8.2.2 private legacy-mapping discovery paths were replaced by the public curated module-mapping resource.

The scientific algorithms, module definitions, statistical procedures, and publication-facing recovery-axis definition were otherwise preserved.

See:

- `provenance/SCRIPT_PROVENANCE.tsv`
- `provenance/RESOURCE_PROVENANCE.tsv`
- `provenance/SOURCE_PROVENANCE.txt`

## Generated participant-level intermediates

Some scripts can generate sample- or participant-level intermediate files when executed with authorized participant data.

Such generated files are not part of the public release and must not be committed to this repository.
