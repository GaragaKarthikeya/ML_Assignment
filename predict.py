"""Predict from a saved model, preserving CSV row order."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train import PolynomialBasis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = joblib.load(args.model)
    data = pd.read_csv(args.test)
    expected = [f"x{i+1}" for i in range(model[0].n_features_in_)]
    if list(data.columns) != expected or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Unexpected columns or non-finite features")
    y = model.predict(data.to_numpy())
    if not np.isfinite(y).all():
        raise ValueError("Non-finite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y}).to_csv(args.output, index=False)
