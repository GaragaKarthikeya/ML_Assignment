# IMT2024073: Polynomial regression assignment

Two separate polynomial regression models predict geothermal net power (var1)
and thermal anomaly scores (var2). All polynomial terms have total degree within
the assignment limits. No non-polynomial prediction model is used.

## Reproduce

Use Python 3.12 and run from this directory:

```powershell
python -m pip install -r requirements.txt
python train.py
python build_report.py
python verify_outputs.py
```

The inputs are in `IMT2024073/`. Outputs are written to `output/`:

- `IMT2024073_pred_var1.csv` and `IMT2024073_pred_var2.csv`: 1,000 rows each,
  a single `y` column, in the same row order as each test file.
- `pdf/IMT2024073_report.pdf`: four-page assignment report.
- `model_search.csv`: development cross-validation scores for every candidate.
- `metrics.json`: selected settings and holdout evaluation.
- `holdout_var1.csv` and `holdout_var2.csv`: actual and predicted holdout targets.
- `models/`: serialized final models, refitted on all training rows.
- `figures/`: report plots.

**Submission format:** checked against `sample_submission.csv`: one `y` column
and 1,000 rows. Filenames match the assignment naming convention.

## Method

Each training dataset is split into 800 development and 200 holdout rows using
seed 42. Hyperparameters are selected by mean MSE from three shuffled development
folds (seed 17). The holdout is not used to select settings. Ridge evaluates raw
monomial and Legendre polynomial bases at every degree 1-10 (var1) and 1-20 (var2),
with alpha from 1e-8 to 1e4 in powers of ten. Lasso evaluates sparse monomial
models at degrees 3, 5, 7, 10 (var1) and 6, 10, 14, 20 (var2), with alpha
0.03, 0.01, 0.003, 0.001. Nonconvergent Lasso candidates are excluded.

All 812 configurations are scored using development folds. Inputs are already
bounded within [-1, 1]; no further normalization is applied. Both bases span
polynomials with all interactions up to the specified total degree. Their
coefficient penalties differ. The intercept is not penalized.

After selection, holdout scores are computed from development-fitted models;
then selected configurations are refitted on all 1,000 rows for test inference.
Hidden test targets are unavailable, so the saved scores are not test scores.
Test var1 inputs have more spread than the training inputs; validation may not
fully represent that distribution difference.

## Results

| Problem | Model | Degree | Alpha | CV MSE | Holdout MSE | Holdout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| var1 | Monomial Lasso | 5 | 0.003 | 0.38929 | 0.37664 | 0.96687 |
| var2 | Monomial ridge | 12 | 0.1 | 0.27312 | 0.24579 | 0.99507 |

The final var1 fit retains 121 of 461 polynomial coefficients. The var2 model
has 454 polynomial coefficients. Results are validation scores, not hidden-test scores.

## Saved-model inference

```powershell
python predict.py --model output/models/var1.joblib --test IMT2024073/IMT2024073_test_var1.csv --output output/reproduced_var1.csv
python predict.py --model output/models/var2.joblib --test IMT2024073/IMT2024073_test_var2.csv --output output/reproduced_var2.csv
```

Only load model files from a trusted source; joblib deserialization executes Python.

The local repository is prepared for GitHub; pushing is handled by the student.
