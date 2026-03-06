# result_analyzer.py - Module 4: Result Analyzer
# Aggregates, compares, and ranks experiment results. Generates comparison tables and JSON/HTML reports.

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class VariationComparison:
    variation_id: str
    model_name: str
    model_family: str
    hyperparameters: Dict[str, Any]
    cv_mean_metrics: Dict[str, float]
    cv_std_metrics: Dict[str, float]
    rank: int
    training_time_sec: float
    status: str
    primary_metric_value: float
    notes: List[str]

@dataclass
class ExperimentReport:
    report_id: str
    objective: str
    task_type: str
    primary_metric: str
    total_variations: int
    best_variation: VariationComparison
    all_variations: List[VariationComparison]
    comparison_table: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]

class ResultAnalyzer:
    """
    Analyzes and compares training results across all experiment variations.
    Produces ranked comparison tables, insights, and exportable reports.

    Usage
    -----
    analyzer = ResultAnalyzer(primary_metric="f1_weighted")
    report = analyzer.analyze(
        objective="Classify emails as spam",
        task_type="classification",
        training_results=[result1, result2, ...],
    )
    """

    def __init__(self, primary_metric: str = "f1_weighted"):
        self.primary_metric = primary_metric

    def analyze(self, objective: str, task_type: str, training_results: list) -> ExperimentReport:
        import uuid
        comparisons = []
        for result in training_results:
            primary_val = result.cv_mean_metrics.get(
                self.primary_metric,
                list(result.cv_mean_metrics.values())[0] if result.cv_mean_metrics else 0.0,
            )
            comparisons.append(VariationComparison(
                variation_id=result.variation_id,
                model_name=result.model_name,
                model_family=result.model_family,
                hyperparameters=result.hyperparameters,
                cv_mean_metrics=result.cv_mean_metrics,
                cv_std_metrics=result.cv_std_metrics,
                rank=0,
                training_time_sec=result.total_training_time_sec,
                status=result.status,
                primary_metric_value=primary_val,
                notes=result.notes,
            ))

        # Rank by primary metric (higher is better except for error metrics)
        lower_is_better = self.primary_metric in ("rmse", "mae", "mape", "davies_bouldin")
        comparisons.sort(key=lambda c: c.primary_metric_value, reverse=not lower_is_better)
        for i, c in enumerate(comparisons):
            c.rank = i + 1

        best = comparisons[0]
        comparison_table = self._build_comparison_table(comparisons)
        insights = self._generate_insights(comparisons, lower_is_better)
        recommendations = self._generate_recommendations(best, comparisons)

        report = ExperimentReport(
            report_id=str(uuid.uuid4())[:8],
            objective=objective,
            task_type=task_type,
            primary_metric=self.primary_metric,
            total_variations=len(comparisons),
            best_variation=best,
            all_variations=comparisons,
            comparison_table=comparison_table,
            insights=insights,
            recommendations=recommendations,
        )
        self._print_report(report)
        return report

    def _build_comparison_table(self, comparisons: List[VariationComparison]) -> List[Dict[str, Any]]:
        rows = []
        for c in comparisons:
            row = {
                "rank": c.rank,
                "variation_id": c.variation_id,
                "model_name": c.model_name,
                "status": c.status,
                "training_time_s": round(c.training_time_sec, 2),
            }
            for metric, val in c.cv_mean_metrics.items():
                row[f"mean_{metric}"] = val
                std = c.cv_std_metrics.get(metric, 0.0)
                row[f"std_{metric}"] = std
            rows.append(row)
        return rows

    def _generate_insights(self, comparisons: List[VariationComparison], lower_is_better: bool) -> List[str]:
        insights = []
        if len(comparisons) < 2:
            return ["Only one variation was tested. Run more experiments for meaningful comparison."]

        best = comparisons[0]
        worst = comparisons[-1]
        spread = abs(best.primary_metric_value - worst.primary_metric_value)

        insights.append(f"Best model '{best.model_name}' achieved {self.primary_metric}={best.primary_metric_value:.4f}.")
        insights.append(f"Performance spread across variations: {spread:.4f} ({spread*100:.1f}% range).")

        if spread < 0.02:
            insights.append("Models perform similarly. Choose the simplest for production.")
        elif spread > 0.10:
            insights.append("Large performance spread detected. Model family has high impact.")

        boost_models = [c for c in comparisons if c.model_family in ("xgboost","random_forest","gradient_boosting")]
        if boost_models:
            avg_boost = sum(c.primary_metric_value for c in boost_models) / len(boost_models)
            non_boost = [c for c in comparisons if c not in boost_models]
            if non_boost:
                avg_non = sum(c.primary_metric_value for c in non_boost) / len(non_boost)
                diff = avg_boost - avg_non
                if abs(diff) > 0.02:
                    direction = "outperform" if diff > 0 else "underperform"
                    insights.append(f"Ensemble/boosting models tend to {direction} linear models by {abs(diff)*100:.1f}%.")

        times = [c.training_time_sec for c in comparisons]
        fastest = comparisons[min(range(len(comparisons)), key=lambda i: times[i])]
        insights.append(f"Fastest training: '{fastest.model_name}' at {fastest.training_time_sec:.2f}s.")
        return insights

    def _generate_recommendations(self, best: VariationComparison, all_comparisons: List[VariationComparison]) -> List[str]:
        recs = [
            f"Deploy '{best.model_name}' (Rank 1, {self.primary_metric}={best.primary_metric_value:.4f}).",
            "Perform additional hyperparameter tuning with Bayesian optimization for the top-ranked model.",
        ]
        if best.model_family in ("xgboost","random_forest","gradient_boosting"):
            recs.append("Consider feature importance analysis to reduce dimensionality.")
        if len(all_comparisons) >= 3:
            recs.append("Build a stacking ensemble from top-3 models for potential further gains.")
        recs.append("Validate the best model on a held-out test set before deployment.")
        return recs

    def _print_report(self, report: ExperimentReport) -> None:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  EXPERIMENT REPORT [{report.report_id}]")
        print(f"  Objective : {report.objective}")
        print(f"  Task Type : {report.task_type}")
        print(f"  Metric    : {report.primary_metric}")
        print(f"  Variations: {report.total_variations}")
        print(sep)
        print(f"{'Rank':<5} {'Variation':<8} {'Model':<30} {'Metric':>10} {'Time(s)':>8} {'Status':<14}")
        print("-" * 70)
        for c in report.all_variations:
            print(f"{c.rank:<5} {c.variation_id:<8} {c.model_name:<30} {c.primary_metric_value:>10.4f} {c.training_time_sec:>8.2f} {c.status:<14}")
        print(sep)
        print("Insights:")
        for ins in report.insights:
            print(f"  - {ins}")
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  + {rec}")
        print(sep + "\n")

    def to_json(self, report: ExperimentReport) -> str:
        def _serialize(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)
        return json.dumps(report.__dict__, default=_serialize, indent=2)

    def to_html_table(self, report: ExperimentReport) -> str:
        rows_html = ""
        for row in report.comparison_table:
            metric_key = f"mean_{report.primary_metric}"
            std_key = f"std_{report.primary_metric}"
            metric_val = row.get(metric_key, row.get("mean_primary_metric", 0.0))
            std_val = row.get(std_key, 0.0)
            badge = " 🏆" if row["rank"] == 1 else ""
            rows_html += f"""
        <tr class="{'best-row' if row['rank'] == 1 else ''}">
          <td>{row['rank']}{badge}</td>
          <td><code>{row['variation_id']}</code></td>
          <td><strong>{row['model_name']}</strong></td>
          <td class="metric">{metric_val:.4f} ± {std_val:.4f}</td>
          <td>{row['training_time_s']:.2f}s</td>
          <td><span class="badge badge-{row['status']}">{row['status']}</span></td>
        </tr>"""
        return f"""<table class="results-table" id="results-{report.report_id}">
      <thead>
        <tr>
          <th>Rank</th>
          <th>ID</th>
          <th>Model</th>
          <th>{report.primary_metric} (mean ± std)</th>
          <th>Train Time</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{rows_html}
      </tbody>
    </table>"""

if __name__ == "__main__":
    # Demo with mock results
    from types import SimpleNamespace
    mock_results = [
        SimpleNamespace(variation_id="V01", model_name="Random Forest", model_family="random_forest",
                        hyperparameters={"n_estimators":100}, cv_mean_metrics={"f1_weighted":0.923,"accuracy":0.931},
                        cv_std_metrics={"f1_weighted":0.012,"accuracy":0.011},
                        total_training_time_sec=4.2, status="completed", notes=["Good performance."]),
        SimpleNamespace(variation_id="V02", model_name="XGBoost", model_family="xgboost",
                        hyperparameters={"n_estimators":200,"learning_rate":0.1},
                        cv_mean_metrics={"f1_weighted":0.911,"accuracy":0.918},
                        cv_std_metrics={"f1_weighted":0.014,"accuracy":0.013},
                        total_training_time_sec=7.1, status="completed", notes=["Excellent performance."]),
        SimpleNamespace(variation_id="V03", model_name="Logistic Regression", model_family="logistic_regression",
                        hyperparameters={"C":1.0}, cv_mean_metrics={"f1_weighted":0.874,"accuracy":0.882},
                        cv_std_metrics={"f1_weighted":0.018,"accuracy":0.017},
                        total_training_time_sec=1.3, status="completed", notes=["Moderate performance."]),
    ]
    analyzer = ResultAnalyzer(primary_metric="f1_weighted")
    report = analyzer.analyze("Classify customer churn", "classification", mock_results)
    print("\nJSON (first 500 chars):")
    print(analyzer.to_json(report)[:500])
