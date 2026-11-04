# Project reports

- `main_experiment_report.pdf`: eight-page experimental report.
- `technical_methods_report.pdf`: revised repository overview: what this repository is,
  how it operates, and how its workflow was used in the project.
- `technical_methods_report.tex`: editable source for the revised report.
- `technical_methods_report_bundle.zip`: revised report and sources, without Python scripts.
- `benchmark_19_report.pdf`: original 22-page report on the existing 19-image set.

These PDFs contain prose, tables, plots and image comparisons, not Python source
listings. The revised technical report uses the current repository filenames and
distinguishes the original full experiment from checks of the refactored package.
The earlier technical report is preserved in `archive/technical_methods_report_v1.pdf`.
Historical filenames in the other original reports can be translated using
`docs/MIGRATION.md`.

Future images you upload to `data/verification/images/` do not change these
historical experiment measurements. Their new outputs go in `outputs/<run-id>/verification/`.

The documentation-only helper `scripts/build_repository_report.py` rebuilds the
revised report from published tables and repository documentation. It requires
the optional reporting dependencies and performs no inference.

