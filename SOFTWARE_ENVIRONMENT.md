# Tested software environment

The public computational companion was audited on 2026-09-10 using Python 3.13.2 and pip 25.1.

The tested package versions are recorded in `requirements.txt`:

- NumPy 2.3.2
- pandas 2.3.3
- SciPy 1.16.1
- statsmodels 0.14.6
- Matplotlib 3.10.5
- Pillow 11.3.0
- openpyxl 3.1.5
- xlrd 2.0.2

All four public script command-line interfaces were successfully invoked with `--help` in this environment.

`pyarrow` and `fastparquet` were not installed in the audited environment. They are therefore not part of the tested dependency set. Parquet and Feather support requires a compatible optional engine if those formats are used.

No historical environment specification was found in the locked private project. The versions above document the tested public-release environment and should not be interpreted as an exact reconstruction of the environment used during every stage of the original analysis.
