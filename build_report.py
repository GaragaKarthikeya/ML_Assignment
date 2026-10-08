"""Build the four-page assignment report from saved experimental results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    search = pd.read_csv(OUT / "model_search.csv")
    charts = OUT / "figures"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for p, ax in enumerate(axes, 1):
        part = search[search.problem == p]
        for (estimator, basis), group in part.groupby(["estimator", "basis"]):
            curve = group.groupby("degree").cv_mse.min()
            ax.plot(curve.index, curve.values, "o-", markersize=3, label=f"{estimator} / {basis}")
        chosen = metrics[f"var{p}"]
        ax.scatter([chosen["degree"]], [chosen["cv_mse"]], marker="*", s=130, color="#b45309", zorder=5)
        ax.set(title=f"var{p}: degree selection", xlabel="Maximum total degree", ylabel="Mean CV MSE", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(charts / "degree_selection.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
    for p in [1, 2]:
        frame = pd.read_csv(OUT / f"holdout_var{p}.csv")
        ax = axes[0, p-1]
        ax.scatter(frame.actual, frame.predicted, s=10, alpha=.65)
        low, high = min(frame.actual.min(), frame.predicted.min()), max(frame.actual.max(), frame.predicted.max())
        ax.plot([low, high], [low, high], color="#b45309", linewidth=1)
        ax.set(title=f"var{p}: held-out predictions", xlabel="Actual y", ylabel="Predicted y")
        ax = axes[1, p-1]
        ax.scatter(frame.predicted, frame.actual-frame.predicted, s=10, alpha=.65)
        ax.axhline(0, color="#b45309", linewidth=1)
        ax.set(title=f"var{p}: residuals", xlabel="Predicted y", ylabel="Actual - predicted")
    fig.tight_layout()
    fig.savefig(charts / "holdout_diagnostics.png", dpi=180)
    plt.close(fig)
    destination = OUT / "pdf"
    destination.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleSmall", parent=styles["Title"], fontSize=22, leading=26, textColor=colors.HexColor("#15324f")))
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 9
    styles["Heading2"].textColor = colors.HexColor("#15324f")
    content = []
    def p(text): content.append(Paragraph(text, styles["BodyText"]))
    def h(text): content.append(Paragraph(text, styles["Heading2"]))
    def table(rows, widths):
        wrapped = [[Paragraph(str(cell), styles["BodyText"]) for cell in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), colors.HexColor("#e8eef4")),
                              ("VALIGN", (0,0),(-1,-1), "TOP"), ("BOTTOMPADDING", (0,0),(-1,-1), 7),
                              ("TOPPADDING", (0,0),(-1,-1), 7),
                              ("LINEBELOW", (0,0),(-1,0), .6, colors.HexColor("#a0b3c6"))]))
        content.extend([t, Spacer(1, 12)])
    content.append(Paragraph("Polynomial Regression for Geothermal Plant Optimization", styles["TitleSmall"]))
    p("Roll number: IMT2024073 | Machine Learning Assignment | 8 October 2026")
    h("1. Objective and data")
    p("The objective is to predict the continuous target y for two personalized datasets using polynomial regression. var1 models the net power score from six operational parameters. var2 models the thermal anomaly score from three spatial coordinates. The problems are trained and evaluated separately.")
    table([["Dataset", "Training", "Test", "Input variables", "Degree limit"],
           ["var1", "1,000 rows", "1,000 rows", "x1 to x6", "10"],
           ["var2", "1,000 rows", "1,000 rows", "x1 to x3", "20"]], [60, 88, 88, 116, 90])
    p("The four input CSVs have the expected columns, finite numeric values, and no missing values. Neither training file contains an exact duplicate row. All input features lie in [-1, 1], so no additional normalization is applied. Test files contain features only; their ground-truth targets are unavailable.")
    h("2. Validation design")
    p("For each problem, a reproducible random split reserves 200 training rows as a holdout and uses 800 rows for model selection (random seed 42). Three shuffled folds within the development set (seed 17) compare candidate models using mean validation MSE. Model selection uses development-fold errors, not test targets or holdout scores.")
    p("After selection, the model is fitted to the 800 development rows and evaluated on the 200 holdout rows. The same selected configuration is then refitted to all 1,000 training rows to produce the final test predictions. The reported holdout results describe models trained on 800 rows, while the submitted predictions use models trained on 1,000 rows.")
    content.append(PageBreak())
    h("3. Polynomial features and regularization")
    p("The model includes all main effects and interaction terms whose total degree is at most d. A term is x1^a1 ... xp^ap, where the nonnegative exponents sum to at most d. Thus, a degree-2 model contains terms such as x1, x1^2 and x1*x2. There are C(p+d, d)-1 nonconstant terms; the intercept is fitted separately.")
    p("Two bases are compared for ridge regression: ordinary monomials and products of Legendre polynomials with the same total-degree constraint. Legendre features span the same polynomial function space but change the numerical conditioning and the regularization penalty. Both bases are evaluated on the original bounded inputs.")
    p("Ridge minimizes squared error plus alpha times the sum of squared coefficients, leaving the intercept unpenalized. It is tested at every degree from 1 to 10 for var1 and 1 to 20 for var2, with 13 logarithmically spaced alpha values from 10^-8 to 10^4. A sparse monomial model is also tested using Lasso, which minimizes squared error divided by 2n plus alpha times the sum of absolute coefficients.")
    p("The Lasso grid uses degrees {3, 5, 7, 10} for var1 and {6, 10, 14, 20} for var2, with alpha values {0.03, 0.01, 0.003, 0.001}. Lasso candidates reaching the iteration limit in a fold are excluded. The combined search covers 812 configurations; alpha has different meanings for Ridge and Lasso and is not directly comparable between them.")
    content.append(Image(str(charts / "degree_selection.png"), width=6.1*inch, height=2.17*inch))
    p("Figure 1. Minimum development CV MSE at each searched degree, after tuning alpha within each model family. Stars mark the selected configurations. Higher degree alone does not guarantee lower validation error.")
    content.append(PageBreak())
    h("4. Selected models and evaluation")
    rows = [["Problem", "Selected model", "Degree / terms", "Alpha", "CV MSE"]]
    for name, m in metrics.items():
        rows.append([name, f"{m['estimator'].capitalize()}, {m['basis']}", f"{m['degree']} / {m['features']}", f"{m['alpha']:g}", f"{m['cv_mse']:.5f}"])
    table(rows, [50, 142, 98, 60, 92])
    rows = [["Problem", "Holdout MSE", "Holdout R2", "Mean baseline MSE"]]
    for name, m in metrics.items():
        rows.append([name, f"{m['holdout_mse']:.5f}", f"{m['holdout_r2']:.5f}", f"{m['baseline_mse']:.5f}"])
    table(rows, [60, 120, 112, 150])
    p("MSE is the mean squared difference between actual and predicted targets; lower is better. R2 measures improvement over predicting the holdout target mean; a score near 1 indicates strong agreement. The baseline shown here predicts the development target mean for every holdout row.")
    content.append(Image(str(charts / "holdout_diagnostics.png"), width=6.1*inch, height=3.52*inch))
    p("Figure 2. Holdout predictions against actual targets and residuals for both problems. The diagonal indicates perfect predictions; the horizontal line indicates zero residual.")
    content.append(PageBreak())
    h("5. Rationale, reproducibility and limitations")
    for name, m in metrics.items():
        p(f"<b>{name}:</b> Degree {m['degree']} with {m['estimator']} regularization was selected because it achieved the lowest mean development CV MSE among the searched candidates ({m['cv_mse']:.5f}). The final fit uses {m['nonzero_terms']} nonzero coefficients out of {m['features']} possible nonconstant terms. The holdout R2 is {m['holdout_r2']:.5f} and MSE is {m['holdout_mse']:.5f}.")
    p("Regularization is necessary because expanding six inputs to degree 10 creates 8,007 nonconstant terms, while expanding three inputs to degree 20 creates 1,770 terms. Both can exceed the available training sample count. The selected degree is a predictive tuning choice, not an estimate of the true physical polynomial degree.")
    p("Validation uses random samples from the training distribution. For var1, test inputs have a larger spread than training inputs, even though both share the same bounds. This distribution difference may change test accuracy. A single holdout of 200 rows also has sampling uncertainty. CV scores are used for selection and may be optimistic after searching many candidates. The hidden test MSE and R2 cannot be computed from the provided files.")
    h("Reproducing the deliverables")
    p("Install requirements.txt, then run <font face='Courier'>python train.py</font> from the project directory. This writes the search results, validation metrics, holdout predictions, fitted models and both final prediction CSVs. Run <font face='Courier'>python build_report.py</font> to regenerate this report and its figures. The repository also contains predict.py for inference with a saved model.")
    table([["Artifact", "Purpose"],
           ["IMT2024073_pred_var1.csv", "1,000 predicted y values for var1"],
           ["IMT2024073_pred_var2.csv", "1,000 predicted y values for var2"],
           ["model_search.csv / metrics.json", "Search evidence and validation results"],
           ["train.py / predict.py / build_report.py", "Training, inference and report generation"]], [200, 242])
    p("The prediction files match the supplied sample_submission.csv: a single header named y and 1,000 prediction rows. Each file preserves the original test-row order. The assignment allows a maximum report length of 4-5 pages; this report has four pages.")
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#a0b3c6"))
        canvas.line(48, 40, 547, 40)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526579"))
        canvas.drawString(48, 27, "IMT2024073 | Polynomial Regression")
        canvas.drawRightString(547, 27, f"Page {doc.page}")
    doc = SimpleDocTemplate(str(destination / "IMT2024073_report.pdf"), pagesize=(595.28, 841.89),
                            leftMargin=60, rightMargin=60, topMargin=45, bottomMargin=55,
                            title="IMT2024073 Polynomial Regression Report", author="IMT2024073")
    doc.build(content, onFirstPage=footer, onLaterPages=footer)
    print(destination / "IMT2024073_report.pdf")

if __name__ == "__main__":
    main()
