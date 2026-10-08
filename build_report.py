"""Regenerate experiment figures and compile the report using pdflatex."""
import json
from pathlib import Path
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    search = pd.read_csv(OUT / "model_search.csv")
    charts = OUT / "figures"
    charts.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for p, ax in enumerate(axes, 1):
        for (estimator, basis), group in search[search.problem == p].groupby(["estimator", "basis"]):
            curve = group.groupby("degree").cv_mse.min()
            ax.plot(curve.index, curve.values, "o-", markersize=3, label=f"{estimator} / {basis}")
        chosen = metrics[f"var{p}"]
        ax.scatter([chosen["degree"]], [chosen["cv_mse"]], marker="*", s=130, color="#b45309", zorder=5)
        ax.set(title=f"var{p}: degree selection", xlabel="Polynomial degree", ylabel="Mean CV MSE", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(charts / "degree_selection.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
    for p in [1, 2]:
        frame = pd.read_csv(OUT / f"holdout_var{p}.csv")
        ax = axes[0, p-1]
        ax.scatter(frame.actual, frame.predicted, s=10, alpha=.65)
        low = min(frame.actual.min(), frame.predicted.min())
        high = max(frame.actual.max(), frame.predicted.max())
        ax.plot([low, high], [low, high], color="#b45309", linewidth=1)
        ax.set(title=f"var{p}: validation predictions", xlabel="Actual y", ylabel="Predicted y")
        ax = axes[1, p-1]
        ax.scatter(frame.predicted, frame.actual-frame.predicted, s=10, alpha=.65)
        ax.axhline(0, color="#b45309", linewidth=1)
        ax.set(title=f"var{p}: residuals", xlabel="Predicted y", ylabel="Actual - predicted")
    fig.tight_layout()
    fig.savefig(charts / "holdout_diagnostics.png", dpi=200)
    plt.close(fig)
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("Install MiKTeX or TeX Live and add pdflatex to PATH.")
    destination = OUT / "pdf"
    destination.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        completed = subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
             f"-output-directory={destination}", "-jobname=IMT2024073_report", "report.tex"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-12000:])
    print(destination / "IMT2024073_report.pdf")

if __name__ == "__main__":
    main()
