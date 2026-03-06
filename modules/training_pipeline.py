# training_pipeline.py - Module 3: Training Pipeline
# Simulates ML model training with cross-validation, epoch tracking, and performance metrics.

from __future__ import annotations
import random, math, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_metric: float
    val_metric: float

@dataclass
class FoldResult:
    fold_id: int
    model_name: str
    train_samples: int
    val_samples: int
    epochs: List[EpochMetrics]
    final_metrics: Dict[str, float]
    training_time_sec: float
    best_epoch: int
    early_stopped: bool

@dataclass
class TrainingResult:
    variation_id: str
    model_name: str
    model_family: str
    hyperparameters: Dict[str, Any]
    fold_results: List[FoldResult]
    cv_mean_metrics: Dict[str, float]
    cv_std_metrics: Dict[str, float]
    total_training_time_sec: float
    status: str
    notes: List[str] = field(default_factory=list)

class TrainingPipeline:
    MODEL_PERFORMANCE_BASE = {
        "logistic_regression": {"accuracy": (0.78, 0.88)},
        "random_forest":       {"accuracy": (0.87, 0.95)},
        "xgboost":             {"accuracy": (0.89, 0.96)},
        "svm":                 {"accuracy": (0.83, 0.92)},
        "mlp":                 {"accuracy": (0.85, 0.93)},
        "linear_regression":   {"r2": (0.65, 0.82)},
        "ridge":               {"r2": (0.72, 0.88)},
        "gradient_boosting":   {"r2": (0.82, 0.94)},
        "lasso":               {"r2": (0.70, 0.86)},
        "svr":                 {"r2": (0.76, 0.90)},
        "kmeans":              {"silhouette": (0.35, 0.65)},
        "dbscan":              {"silhouette": (0.40, 0.70)},
        "agglomerative":       {"silhouette": (0.38, 0.65)},
        "gmm":                 {"silhouette": (0.42, 0.68)},
        "kmeans_plus":         {"silhouette": (0.45, 0.72)},
    }

    def __init__(self, random_seed=42, verbose=True):
        self.random_seed = random_seed
        self.verbose = verbose
        random.seed(random_seed)

    def run(self, variation_id, model_name, model_family, hyperparameters,
            task_type="classification", n_folds=5, n_epochs=50,
            early_stopping_patience=10, dataset_size=10000):
        if self.verbose:
            print(f"[{variation_id}] Training: {model_name} | {task_type} | {n_folds} folds")
        fold_results = []
        start_total = time.time()
        for fold_id in range(1, n_folds + 1):
            fold = self._run_fold(fold_id, model_name, model_family, hyperparameters,
                                  task_type, n_epochs, early_stopping_patience, dataset_size)
            fold_results.append(fold)
            if self.verbose:
                m = " | ".join(f"{k}={v:.4f}" for k, v in fold.final_metrics.items())
                print(f"  Fold {fold_id}: {m} | {fold.training_time_sec:.2f}s")
        total_time = time.time() - start_total
        cv_mean, cv_std = self._aggregate_cv_metrics(fold_results)
        status = "early_stopped" if any(f.early_stopped for f in fold_results) else "completed"
        result = TrainingResult(
            variation_id=variation_id, model_name=model_name, model_family=model_family,
            hyperparameters=hyperparameters, fold_results=fold_results,
            cv_mean_metrics=cv_mean, cv_std_metrics=cv_std,
            total_training_time_sec=round(total_time, 3), status=status,
            notes=self._generate_notes(model_family, cv_mean),
        )
        if self.verbose:
            self._print_summary(result)
        return result

    def _run_fold(self, fold_id, model_name, model_family, hyperparameters,
                  task_type, n_epochs, patience, dataset_size):
        random.seed(self.random_seed + fold_id * 17)
        train_size = int(dataset_size * 0.8)
        base = self.MODEL_PERFORMANCE_BASE.get(model_family, {"accuracy": (0.80, 0.92)})
        lo, hi = list(base.values())[0]
        target = lo + (hi - lo) * random.uniform(0.6, 1.0)
        epochs = self._simulate_curve(n_epochs, target, patience)
        best_epoch = max(range(len(epochs)), key=lambda i: epochs[i].val_metric) + 1
        sim_time = round(random.uniform(0.3, 2.5), 3)
        return FoldResult(
            fold_id=fold_id, model_name=model_name,
            train_samples=train_size, val_samples=dataset_size - train_size,
            epochs=epochs, final_metrics=self._compute_metrics(task_type, target, epochs),
            training_time_sec=sim_time, best_epoch=best_epoch,
            early_stopped=len(epochs) < n_epochs,
        )

    def _simulate_curve(self, n_epochs, target, patience):
        epochs, best_val, no_improve = [], 0.0, 0
        for e in range(1, n_epochs + 1):
            p = e / n_epochs
            tm = min(0.9999, target * (1 - math.exp(-5 * p)) + random.gauss(0, 0.008))
            vm = min(0.9999, target * (1 - math.exp(-4 * p)) * 0.97 + random.gauss(0, 0.012))
            epochs.append(EpochMetrics(
                epoch=e,
                train_loss=round(max(0.001, 1.0 - tm + random.gauss(0, 0.01)), 5),
                val_loss=round(max(0.001, 1.0 - vm + random.gauss(0, 0.015)), 5),
                train_metric=round(max(0.0, tm), 5),
                val_metric=round(max(0.0, vm), 5),
            ))
            if vm > best_val + 0.001:
                best_val, no_improve = vm, 0
            else:
                no_improve += 1
            if no_improve >= patience and e >= 10:
                break
        return epochs

    def _compute_metrics(self, task_type, target, epochs):
        bv = max(e.val_metric for e in epochs)
        if task_type == "classification":
            return {
                "accuracy": round(min(0.9999, bv * random.uniform(0.98, 1.02)), 4),
                "f1_weighted": round(min(0.9999, bv * random.uniform(0.96, 1.00)), 4),
                "precision": round(min(0.9999, bv * random.uniform(0.95, 1.01)), 4),
                "recall": round(min(0.9999, bv * random.uniform(0.94, 1.00)), 4),
                "roc_auc": round(min(0.9999, bv * random.uniform(1.01, 1.06)), 4),
            }
        elif task_type == "regression":
            rmse = round(max(0.001, (1 - bv) * random.uniform(0.4, 0.6)), 4)
            return {"r2_score": round(min(0.9999, bv), 4), "rmse": rmse,
                    "mae": round(max(0.001, rmse * random.uniform(0.7, 0.9)), 4),
                    "mape": round(rmse * random.uniform(8, 15), 4)}
        elif task_type == "clustering":
            return {"silhouette_score": round(random.uniform(0.35, 0.72), 4),
                    "davies_bouldin": round(random.uniform(0.6, 1.8), 4),
                    "calinski_harabasz": round(random.uniform(120, 500), 2)}
        return {"primary_metric": round(bv, 4)}

    def _aggregate_cv_metrics(self, fold_results):
        if not fold_results:
            return {}, {}
        means, stds = {}, {}
        for key in fold_results[0].final_metrics:
            vals = [f.final_metrics.get(key, 0.0) for f in fold_results]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean)**2 for v in vals) / max(1, len(vals) - 1))
            means[key] = round(mean, 4)
            stds[key] = round(std, 4)
        return means, stds

    def _generate_notes(self, model_family, cv_mean):
        primary = list(cv_mean.values())[0] if cv_mean else 0.0
        if primary > 0.93:
            return ["Excellent performance. Consider further hyperparameter refinement."]
        elif primary > 0.85:
            return ["Good performance. Ensemble methods may further improve results."]
        return ["Moderate performance. Consider feature engineering or more data."]

    def _print_summary(self, result):
        print(f"  CV Results | Status: {result.status}")
        for m, mean in result.cv_mean_metrics.items():
            std = result.cv_std_metrics.get(m, 0.0)
            print(f"    {m}: {mean:.4f} +/- {std:.4f}")

if __name__ == "__main__":
    TrainingPipeline().run(
        variation_id="V01", model_name="Random Forest", model_family="random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        task_type="classification", n_folds=5, dataset_size=10000,
    )
