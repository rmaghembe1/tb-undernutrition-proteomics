# Reviewer-revision computational environment

The reviewer-requested revision analyses were audited under:

- Python 3.13.2
- NumPy 2.3.2
- pandas 2.3.3
- SciPy 1.16.1
- statsmodels 0.14.6
- Matplotlib 3.10.5

Install dependencies with:

    python3 -m pip install -r requirements-reviewer-revision.txt

The four scripts in `scripts/reviewer_revision/` are byte-identical copies
of the authenticated scripts used during the manuscript revision.

They execute at module level and should be run as standalone scripts.
Do not import them as Python modules.
