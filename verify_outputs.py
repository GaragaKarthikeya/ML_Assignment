"""Check submission schema and independent saved-model inference."""
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent

def main():
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    search = pd.read_csv(ROOT / "output/model_search.csv")
    metrics = json.loads((ROOT / "output/metrics.json").read_text())
    for p in [1, 2]:
        prediction_path = ROOT / f"output/IMT2024073_pred_var{p}.csv"
        predictions = pd.read_csv(prediction_path)
        inputs = ROOT / f"IMT2024073/IMT2024073_test_var{p}.csv"
        assert list(predictions.columns) == list(sample.columns) == ["y"]
        assert len(predictions) == len(sample) == len(pd.read_csv(inputs)) == 1000
        assert np.isfinite(predictions.to_numpy()).all()
        chosen = metrics[f"var{p}"]
        assert np.isclose(chosen["cv_mse"], search[search.problem == p].cv_mse.min())
        model_path = ROOT / f"output/models/var{p}.joblib"
        model = joblib.load(model_path)
        if chosen["estimator"] == "lasso":
            assert model[-1].n_iter_ < model[-1].max_iter
        recreated = ROOT / f"tmp/reproduced_var{p}.csv"
        subprocess.run([sys.executable, str(ROOT / "predict.py"), "--model", str(model_path),
                        "--test", str(inputs), "--output", str(recreated)], check=True)
        np.testing.assert_allclose(predictions.y, pd.read_csv(recreated).y, rtol=1e-10, atol=1e-10)
        print(f"var{p}: schema, row count, finite values, CV selection and saved-model inference passed")

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
