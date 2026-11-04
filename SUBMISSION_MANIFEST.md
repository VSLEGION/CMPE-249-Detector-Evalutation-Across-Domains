# Submission packages

`submission/cmpe249_track_a_submission.zip` is the main upload. It contains one
top-level project directory with:

- The primary eight-page report and two clearly labeled supplementary PDFs.
- Report sources/figure dependencies, configuration and complete experiment code.
- Original metric tables, raw timing samples, logs, model metadata and diagnostics.
- The 19 input images, scientific figures and representative comparisons.
- A small ground-truth/prediction pair for exact metric reproduction.
- Checks and fresh qualitative verification evidence from this copied folder.

`submission/report_ready_bundle.zip` is the compact report/support bundle. It
omits raw source images and the detailed 22-page image supplement; use it for
quick review together with the primary report.

`submission/prediction_evidence_bundle.zip` is optional, larger evidence containing
the original full predictions/converted GT. It is not nested inside the main ZIP.

The archives exclude `.git/`, environments, weights, large datasets,
`configs/local.json`, `outputs/`, caches, archived superseded material and other ZIPs.
They also exclude unrelated GPU process listings. The current private local override
continues to work on this machine but is not included in the upload.

`submission/archive_verification.json` records exact member names, sizes and hashes.
`results/submission_checks/validation.json` records actual checks and PDF page counts.
All output paths in the handoff point to persistent project files.

Rebuild/check using the existing dtect interpreter and the helper described in
`docs/REPRODUCE.md`. Building packages is a local operation; it never uploads
coursework, pushes commits or contacts another person.
