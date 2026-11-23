# Reviewer-requested revision analyses

These scripts document the post hoc analyses performed in response to
Reviewer 2 for manuscript 2018164.

The files under `scripts/reviewer_revision/` are byte-for-byte unchanged
from the authenticated revision workflow.

No participant-level processed proteomics or clinical data are redistributed
with this code release. Re-execution requires appropriately authorized source
inputs and the relevant frozen intermediate tables.

## Interpretation boundary

All represented analyses were reviewer-requested/post hoc.

Null BMI-by-time interactions indicate failure to detect differential
trajectories at the available precision; they do not establish equivalence.

The 36-protein construction is a treatment-associated nutritional-metabolic
PC1 summary and not a validated unidimensional recovery scale.

## Randomness

- F14_R5A_R1_analysis.py: explicit seed 20261122.
- F14_R5A_R2_addendum.py: explicit seed 20261122.
- F14_R5B1_analysis.py: no stochastic calls.
- F14_R5B2_analysis.py: no stochastic calls.

The earlier public v1.0.0 workflow contains a separate historical bootstrap
implementation using seed 20260715.

## Invocation contracts

### F14_R5A_R1_analysis.py

    python3 scripts/reviewer_revision/F14_R5A_R1_analysis.py \
      BASE SAMPLE LONG S13 S14 S16 S17 OUT AXIS_EXISTING

### F14_R5A_R2_addendum.py

    python3 scripts/reviewer_revision/F14_R5A_R2_addendum.py \
      BASE LONG S13 S14 R5A OUT

### F14_R5B1_analysis.py

    python3 scripts/reviewer_revision/F14_R5B1_analysis.py \
      LONG MISSAUDIT MIXFIT MIXINT MIXJOINT OUT

### F14_R5B2_analysis.py

    python3 scripts/reviewer_revision/F14_R5B2_analysis.py \
      LONG SUPPORT STRICTINT FITSTATUS FROZENINT CALIB CALSUM OUT

## Data boundary

Public source-study mass-spectrometry records are associated with:

- ProteomeXchange/PRIDE: PXD040546
- MassIVE: MSV000091392

The collaborator-supplied processed abundance matrix, clinical metadata,
sample linkage, and participant-level analysis tables are not redistributed.

## Release lineage

Public v1.0.0 commit:

    243e65798af9ba59f1b498215e862ba330ead2c4

Existing Zenodo DOI:

    10.5281/zenodo.22692544

The Reviewer 2 scripts postdate v1.0.0 and are documented as an explicit
revision layer rather than being represented as part of the earlier release.
