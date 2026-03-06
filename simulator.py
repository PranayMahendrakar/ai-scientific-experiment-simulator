# simulator.py - AI Scientific Experiment Simulator - Main Orchestrator
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modules.experiment_planner import ExperimentPlanner
from modules.model_selector import ModelSelector
from modules.training_pipeline import TrainingPipeline
from modules.result_analyzer import ResultAnalyzer


class AIExperimentSimulator:
    """End-to-end AI Scientific Experiment Simulator.
    
    Orchestrates 4 modules:
      1. ExperimentPlanner  - Designs variations from research objective
      2. ModelSelector      - Selects optimal models per variation
      3. TrainingPipeline   - Simulates training with cross-validation
      4. ResultAnalyzer     - Compares and ranks all results
    """

    def __init__(self, max_variations=5, n_folds=5, n_epochs=50,
                 random_seed=42, verbose=True, output_dir="reports"):
        self.max_variations = max_variations
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.planner = ExperimentPlanner(max_variations=max_variations, random_seed=random_seed)
        self.selector = ModelSelector(random_seed=random_seed)
        self.pipeline = TrainingPipeline(random_seed=random_seed, verbose=verbose)

    def run(self, objective: str) -> dict:
        print("\n" + "="*70)
        print("  AI SCIENTIFIC EXPERIMENT SIMULATOR")
        print("="*70)
        print(f"  Objective: {objective}")
        print("="*70)

        print("\n[Step 1/4] Designing experiment plan...")
        plan = self.planner.design(objective)

        print("\n[Step 2/4] Selecting optimal models...")
        selection = self.selector.select(task_type=plan.task_type, objective=objective)

        print(f"\n[Step 3/4] Running {len(plan.variations)} training variations...")
        training_results = []
        for variation in plan.variations:
            model_cfg = next(
                (m for m in selection.ranked_models if m.model_family == variation.model_family),
                selection.ranked_models[0] if selection.ranked_models else None,
            )
            hyperparameters = {}
            if model_cfg:
                hyperparameters = {hp.param_name: hp.default for hp in model_cfg.hyperparameter_grids}
            result = self.pipeline.run(
                variation_id=variation.variation_id,
                model_name=variation.name,
                model_family=variation.model_family,
                hyperparameters=hyperparameters,
                task_type=plan.task_type,
                n_folds=plan.evaluation_strategy.n_splits if plan.evaluation_strategy.method != "hold_out" else 1,
                n_epochs=self.n_epochs,
                dataset_size=plan.dataset_profile.estimated_samples,
            )
            training_results.append(result)

        print("\n[Step 4/4] Analyzing and comparing results...")
        analyzer = ResultAnalyzer(primary_metric=plan.primary_metric)
        report = analyzer.analyze(
            objective=objective, task_type=plan.task_type, training_results=training_results
        )
        self._export_report(report, analyzer)
        return {"plan": plan, "selection": selection, "training_results": training_results, "report": report}

    def _export_report(self, report, analyzer) -> None:
        rid = report.report_id
        json_path = self.output_dir / f"report_{rid}.json"
        with open(json_path, "w") as f:
            f.write(analyzer.to_json(report))
        print(f"  [Export] JSON  => {json_path}")

        html_path = self.output_dir / f"report_{rid}.html"
        with open(html_path, "w") as f:
            f.write(self._build_html(report, analyzer))
        print(f"  [Export] HTML  => {html_path}")

    def _build_html(self, report, analyzer) -> str:
        table = analyzer.to_html_table(report)
        insights = "".join(f"<li>{i}</li>" for i in report.insights)
        recs = "".join(f"<li>{r}</li>" for r in report.recommendations)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Experiment Report {report.report_id}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:24px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}}
.results-table{{width:100%;border-collapse:collapse;margin:16px 0}}
.results-table th{{background:#161b22;color:#58a6ff;padding:10px;text-align:left;border-bottom:2px solid #30363d}}
.results-table td{{padding:10px;border-bottom:1px solid #21262d}}.best-row{{background:#1a2f1a}}
.metric{{font-weight:bold;color:#3fb950}}.badge{{padding:2px 8px;border-radius:12px;font-size:12px}}
.badge-completed{{background:#196f3d;color:#3fb950}}.badge-early_stopped{{background:#7d4e00;color:#e3b341}}
ul{{padding-left:20px}}li{{margin:6px 0}}code{{background:#161b22;padding:2px 6px;border-radius:4px;font-family:monospace}}
</style></head>
<body>
<h1>Experiment Report <code>{report.report_id}</code></h1>
<div class="meta">
  <p><strong>Objective:</strong> {report.objective}</p>
  <p><strong>Task:</strong> {report.task_type} | <strong>Metric:</strong> {report.primary_metric}</p>
  <p><strong>Variations:</strong> {report.total_variations} | <strong>Best:</strong> {report.best_variation.model_name}</p>
</div>
<h2>Results Comparison</h2>{table}
<h2>Insights</h2><ul>{insights}</ul>
<h2>Recommendations</h2><ul>{recs}</ul>
<hr><p style="color:#8b949e;font-size:12px">Generated by AI Scientific Experiment Simulator</p>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="AI Scientific Experiment Simulator")
    parser.add_argument("--objective", "-o", default="Classify emails as spam or not spam",
                        help="Research objective (free text)")
    parser.add_argument("--variations", "-v", type=int, default=5)
    parser.add_argument("--folds", "-k", type=int, default=5)
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output", default="reports")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    sim = AIExperimentSimulator(
        max_variations=args.variations, n_folds=args.folds, n_epochs=args.epochs,
        random_seed=args.seed, verbose=not args.quiet, output_dir=args.output,
    )
    sim.run(args.objective)
    print("\nDone! Reports saved to:", args.output)


if __name__ == "__main__":
    main()
