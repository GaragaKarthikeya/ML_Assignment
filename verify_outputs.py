"""Audit submission format, model degree, validation scores and inference."""
import json
from pathlib import Path
import subprocess
import sys
import math
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent

def main():
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    search = pd.read_csv(ROOT / "output/model_search.csv")
    metrics = json.loads((ROOT / "output/metrics.json").read_text())
    for p in [1, 2]:
        prediction_path = ROOT / f"output/IMT2024073_pred_var{p}.csv"
        predictions = pd.read_csv(prediction_path)
        inputs = ROOT / f"IMT2024073/IMT2024073_test_var{p}.csv"
        training = pd.read_csv(ROOT / f"IMT2024073/IMT2024073_train_var{p}.csv")
        test = pd.read_csv(inputs)
        n_features = 6 if p == 1 else 3
        columns = [f"x{i}" for i in range(1, n_features + 1)]
        assert list(training.columns) == columns + ["y"]
        assert list(test.columns) == columns
        assert np.isfinite(training.to_numpy()).all()
        assert np.isfinite(test.to_numpy()).all()
        assert list(predictions.columns) == list(sample.columns) == ["y"]
        assert len(predictions) == len(sample) == len(pd.read_csv(inputs)) == 1000
        assert np.isfinite(predictions.to_numpy()).all()
        chosen = metrics[f"var{p}"]
        assert np.isclose(chosen["cv_mse"], search[search.problem == p].cv_mse.min())
        model_path = ROOT / f"output/models/var{p}.joblib"
        model = joblib.load(model_path)
        assert 1 <= chosen["degree"] <= (10 if p == 1 else 20)
        assert model[0].degree == chosen["degree"]
        assert model[0].basis == chosen["basis"]
        assert model[-1].alpha == chosen["alpha"]
        assert chosen["features"] == math.comb(n_features + chosen["degree"], chosen["degree"]) - 1
        powers = model[0].powers_
        assert np.all((powers.sum(axis=1) >= 1) & (powers.sum(axis=1) <= chosen["degree"]))
        assert np.unique(powers, axis=0).shape[0] == chosen["features"]
        assert np.count_nonzero(model[-1].coef_) == chosen["nonzero_terms"]
        if chosen["estimator"] == "lasso":
            assert model[-1].n_iter_ < model[-1].max_iter
        # Rebuild the validation fit from the raw training data, independently
        # of the saved holdout predictions and final model coefficients.
        development, holdout = train_test_split(np.arange(len(training)), test_size=.2, random_state=42)
        stored = pd.read_csv(ROOT / f"output/holdout_var{p}.csv")
        np.testing.assert_array_equal(stored.row_index, holdout)
        np.testing.assert_allclose(stored.actual, training.y.to_numpy()[holdout], rtol=1e-14, atol=1e-14)
        assert not set(development).intersection(holdout)
        fitted = clone(model).fit(training[columns].to_numpy()[development], training.y.to_numpy()[development])
        holdout_predictions = fitted.predict(training[columns].to_numpy()[holdout])
        np.testing.assert_allclose(stored.predicted, holdout_predictions, rtol=1e-10, atol=1e-10)
        assert np.isclose(mean_squared_error(stored.actual, stored.predicted), chosen["holdout_mse"])
        assert np.isclose(r2_score(stored.actual, stored.predicted), chosen["holdout_r2"])
        # Evaluate monomials directly rather than through the custom transformer.
        if chosen["basis"] == "monomial":
            direct = np.ones((len(test), len(powers)))
            for j in range(n_features):
                direct *= test.to_numpy()[:, j, None] ** powers[None, :, j]
            np.testing.assert_allclose(direct @ model[-1].coef_ + model[-1].intercept_, predictions.y,
                                       rtol=1e-10, atol=1e-10)
        recreated = ROOT / f"tmp/reproduced_var{p}.csv"
        subprocess.run([sys.executable, str(ROOT / "predict.py"), "--model", str(model_path),
                        "--test", str(inputs), "--output", str(recreated)], check=True)
        np.testing.assert_allclose(predictions.y, pd.read_csv(recreated).y, rtol=1e-10, atol=1e-10)
        print(f"var{p}: inputs, submission, polynomial degree, CV selection, validation refit, metrics and inference passed")

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
