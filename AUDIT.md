# Assignment audit - 8 October 2026

I reread all three assignment screenshots and reviewed the complete training,
inference, verification and report-generation code, the LaTeX source and all
three rendered PDF pages. This audit covers the supplied requirements; no
additional marking rubric or hidden test targets were available.

## Requirements

| Requirement | Finding |
| --- | --- |
| Personalized datasets | Both problems use the four IMT2024073 CSVs. |
| Separate models | var1 and var2 are trained independently. |
| Polynomial regression only | Both final models are explicit polynomial expansions with linear coefficient fitting and regularization. |
| Total-degree limits | var1 is degree 5, within 10; var2 is degree 12, within 20. All term exponent sums were checked. |
| Degree choice and rationale | The report gives the search procedure, selected degrees, comparisons and regularization choices. |
| MSE and R2 | Both are reported and independently recalculated on reserved validation samples. They are not presented as hidden-test results. |
| Concise PDF, maximum 4-5 pages | Three-page pdflatex report; every page visually checked. |
| Prediction naming | IMT2024073_pred_var1.csv and IMT2024073_pred_var2.csv. |
| Sample submission format | Each CSV matches the sample: one y column and 1,000 rows, without an extra index. |
| Training and inference code | Complete code is tracked in the local Git repository. |
| GitHub repository | Still requires the student's push and repository-link submission. |

## Verification

- Repeated the entire 812-configuration model search in a separate output folder.
  Search results, holdout CSVs and both submission CSVs reproduced byte for byte;
  metrics reproduced exactly.
- Recreated both validation fits from the raw training data, checked the split
  indices and recalculated MSE and R2 independently.
- Confirmed the reserved validation indices do not occur in the development set.
  Polynomial expansion has no data-dependent fitted statistics; centering for
  Ridge is computed within each training fold. Selection uses development CV.
- Evaluated the final polynomials directly from exponents and coefficients and
  compared against the submission predictions in original test-row order.
- Repeated inference through the separate predict.py process and saved models.
- Compared the custom Ridge path solver against scikit-learn's independent SVD
  solver in both its primal and dual branches, including both polynomial bases.
- Confirmed the selected final Lasso converged. Some unsuccessful high-degree
  Lasso search candidates were excluded as described in the report.
- Checked installed dependency versions against requirements.txt; pip check
  found no broken dependencies.
- Built the report from a clean local Git clone with pdflatex. There were no
  compilation errors, layout warnings or missing figures.
- Checked that the submission ZIP contains the current PDF and CSVs unchanged.

## Remaining submission steps

Push the local repository to GitHub and provide its URL as required by the
course. Submit the PDF and both prediction CSVs through the course's submission
process. The ZIP is a convenience bundle, not a replacement for separately
uploading files if the portal requests them.

No substantive model, calculation, format or report errors were found. This
does not guarantee a full grade: hidden-test accuracy, grading thresholds and
the instructor's evaluation are unavailable. The greater spread of var1 test
inputs may affect hidden-test results; this limitation is stated in the report.
