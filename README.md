# projSwing

Tennis Match Charting Project analysis and data-quality tooling.

## Notebooks

- [`player_qc.ipynb`](player_qc.ipynb) is the reusable preprocessing entry point.
  Set `PLAYER`, the earliest plausible match date, and the Google Drive paths in
  its configuration cell. It records exclusions and saves clean, per-player CSV
  artifacts under `qc/<player_slug>/`.
- [`alcarazAnalysis.ipynb`](alcarazAnalysis.ipynb) is the existing Carlos
  Alcaraz proof-of-concept analysis, renamed from `projSwing.ipynb`. Its analysis
  code is unchanged and is not yet wired to the QC artifacts.

The validation and artifact I/O implementation lives in [`data_qc.py`](data_qc.py).