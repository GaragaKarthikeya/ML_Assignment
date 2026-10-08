"""Select polynomial regressions on development CV; evaluate once on a holdout."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits


class PolynomialBasis(TransformerMixin, BaseEstimator):
    """All interactions of total degree <= degree; raw or Legendre polynomials."""
    def __init__(self, degree=1, basis="monomial"):
        self.degree = degree
        self.basis = basis

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        self.powers_ = PolynomialFeatures(self.degree, include_bias=False).fit(X).powers_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.basis == "monomial":
            return PolynomialFeatures(self.degree, include_bias=False).fit_transform(X)
        tables = [np.polynomial.legendre.legvander(X[:, j], self.degree)
                  for j in range(self.n_features_in_)]
        result = np.ones((len(X), len(self.powers_)))
        for j, table in enumerate(tables):
            result *= table[:, self.powers_[:, j]]
        return result


def ridge_path_predict(Z, y, V, alphas):
    """One eigen decomposition per fold, reused across all ridge strengths."""
    mean = Z.mean(axis=0)
    A, B = Z - mean, V - mean
    ym = y.mean()
    yc = y - ym
    if A.shape[1] <= A.shape[0]:
        vals, vecs = np.linalg.eigh(A.T @ A)
        vals = np.maximum(vals, 0)
        rhs = vecs.T @ (A.T @ yc)
        return (B @ vecs) @ (rhs[:, None] / (vals[:, None] + alphas)) + ym
    vals, vecs = np.linalg.eigh(A @ A.T)
    vals = np.maximum(vals, 0)
    rhs = vecs.T @ yc
    return ((B @ A.T) @ vecs) @ (rhs[:, None] / (vals[:, None] + alphas)) + ym


def run(data_dir, output_dir, roll):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    alphas = np.logspace(-8, 4, 13)
    summaries = {}
    all_rows = []
    for problem, max_degree in [(1, 10), (2, 20)]:
        train = pd.read_csv(data_dir / f"{roll}_train_var{problem}.csv")
        test = pd.read_csv(data_dir / f"{roll}_test_var{problem}.csv")
        columns = [f"x{i}" for i in range(1, 7 if problem == 1 else 4)]
        if list(train.columns) != columns + ["y"] or list(test.columns) != columns:
            raise ValueError("Unexpected input columns")
        if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
            raise ValueError("Non-finite input")
        X, y = train[columns].to_numpy(), train.y.to_numpy()
        dev, holdout = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
        folds = list(KFold(3, shuffle=True, random_state=17).split(dev))
        rows = []
        for basis in ["monomial", "legendre"]:
            for degree in range(1, max_degree + 1):
                Z = PolynomialBasis(degree, basis).fit_transform(X[dev])
                errors = []
                for a, b in folds:
                    pred = ridge_path_predict(Z[a], y[dev[a]], Z[b], alphas)
                    errors.append(np.mean((pred - y[dev[b], None]) ** 2, axis=0))
                errors = np.array(errors)
                for k, alpha in enumerate(alphas):
                    rows.append(dict(problem=problem, estimator="ridge", basis=basis, degree=degree,
                                     alpha=float(alpha), features=Z.shape[1],
                                     cv_mse=float(errors[:, k].mean()),
                                     cv_mse_std=float(errors[:, k].std())))
                best = min(rows, key=lambda r: r["cv_mse"])
                print(f"var{problem} {basis} degree {degree}: best so far {best}", flush=True)
        sparse_degrees = [3, 5, 7, 10] if problem == 1 else [6, 10, 14, 20]
        for degree in sparse_degrees:
            Z = PolynomialBasis(degree, "monomial").fit_transform(X[dev])
            sparse_alphas = [0.03, 0.01, 0.003, 0.001]
            errors = np.zeros((3, len(sparse_alphas)))
            for fold, (a, b) in enumerate(folds):
                sparse = Lasso(warm_start=True, max_iter=10000, tol=1e-5)
                for k, alpha in enumerate(sparse_alphas):
                    sparse.set_params(alpha=alpha)
                    sparse.fit(Z[a], y[dev[a]])
                    if sparse.n_iter_ == sparse.max_iter:
                        errors[fold, k] = np.inf
                    else:
                        errors[fold, k] = mean_squared_error(y[dev[b]], sparse.predict(Z[b]))
            for k, alpha in enumerate(sparse_alphas):
                rows.append(dict(problem=problem, estimator="lasso", basis="monomial",
                                 degree=degree, alpha=alpha, features=Z.shape[1],
                                 cv_mse=float(errors[:, k].mean()),
                                 cv_mse_std=float(errors[:, k].std()) if np.isfinite(errors[:, k]).all() else None))
            print(f"var{problem} lasso degree {degree}: {errors.mean(axis=0)}", flush=True)
        chosen = min(rows, key=lambda r: r["cv_mse"])
        def model():
            regressor = (Ridge(alpha=chosen["alpha"], solver="cholesky")
                         if chosen["estimator"] == "ridge" else
                         Lasso(alpha=chosen["alpha"], max_iter=30000, tol=1e-6))
            return make_pipeline(PolynomialBasis(chosen["degree"], chosen["basis"]), regressor)
        fitted = model().fit(X[dev], y[dev])
        hp = fitted.predict(X[holdout])
        baseline = np.full(len(holdout), y[dev].mean())
        summary = dict(chosen, train_rows=len(train), test_rows=len(test), dev_rows=len(dev),
                       holdout_rows=len(holdout), holdout_mse=mean_squared_error(y[holdout], hp),
                       holdout_r2=r2_score(y[holdout], hp),
                       baseline_mse=mean_squared_error(y[holdout], baseline),
                       baseline_r2=r2_score(y[holdout], baseline),
                       duplicate_train_rows=int(train.duplicated().sum()),
                       train_feature_range=[float(X.min()), float(X.max())],
                       test_feature_range=[float(test.to_numpy().min()), float(test.to_numpy().max())])
        pd.DataFrame({"row_index": holdout, "actual": y[holdout], "predicted": hp}).to_csv(
            output_dir / f"holdout_var{problem}.csv", index=False)
        final = model().fit(X, y)
        if chosen["estimator"] == "lasso" and final[-1].n_iter_ == final[-1].max_iter:
            raise RuntimeError("Selected final Lasso model did not converge")
        summary["nonzero_terms"] = int(np.count_nonzero(final[-1].coef_))
        prediction = final.predict(test.to_numpy())
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions")
        pd.DataFrame({"y": prediction}).to_csv(output_dir / f"{roll}_pred_var{problem}.csv", index=False)
        joblib.dump(final, output_dir / "models" / f"var{problem}.joblib")
        summaries[f"var{problem}"] = summary
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(output_dir / "model_search.csv", index=False)
        (output_dir / "metrics.json").write_text(json.dumps(summaries, indent=2))
        print("SELECTED", json.dumps(summary), flush=True)
    return summaries


if __name__ == "__main__":
    # A stable import name keeps serialized custom transformers usable for inference.
    from train import run as imported_run
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("IMT2024073"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--roll", default="IMT2024073")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        imported_run(args.data_dir, args.output_dir, args.roll)
