# model_selector.py - Module 2: Model Selector
# Maps task types to candidate ML models with configurations and hyperparameter grids.

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class HyperparameterGrid:
    param_name: str
    param_type: str  # 'int', 'float', 'categorical'
    values: List[Any]
    default: Any

@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    model_family: str
    library: str  # 'sklearn', 'xgboost', 'lightgbm', 'torch', 'tensorflow'
    task_types: List[str]
    hyperparameter_grids: List[HyperparameterGrid]
    complexity_score: float  # 1-10 (1=simple, 10=complex)
    interpretability_score: float  # 1-10 (10=fully interpretable)
    min_samples_recommended: int
    supports_feature_importance: bool
    notes: str = ""

@dataclass
class ModelSelectionResult:
    task_type: str
    objective: str
    ranked_models: List[ModelConfig]
    primary_recommendation: ModelConfig
    rationale: str
    hyperparameter_summary: Dict[str, Dict[str, Any]]

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY: List[ModelConfig] = [
    ModelConfig(
        model_id="lr_clf", model_name="Logistic Regression",
        model_family="logistic_regression", library="sklearn",
        task_types=["classification"],
        hyperparameter_grids=[
            HyperparameterGrid("C", "float", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0),
            HyperparameterGrid("penalty", "categorical", ["l2", "l1", "elasticnet"], "l2"),
            HyperparameterGrid("max_iter", "int", [100, 500, 1000], 100),
            HyperparameterGrid("solver", "categorical", ["lbfgs", "saga", "liblinear"], "lbfgs"),
        ],
        complexity_score=2.0, interpretability_score=9.0,
        min_samples_recommended=500, supports_feature_importance=True,
        notes="Good baseline for linearly separable data. Fast training.",
    ),
    ModelConfig(
        model_id="rf_clf", model_name="Random Forest Classifier",
        model_family="random_forest", library="sklearn",
        task_types=["classification"],
        hyperparameter_grids=[
            HyperparameterGrid("n_estimators", "int", [50, 100, 200, 300], 100),
            HyperparameterGrid("max_depth", "int", [None, 5, 10, 20, 30], None),
            HyperparameterGrid("min_samples_split", "int", [2, 5, 10], 2),
            HyperparameterGrid("min_samples_leaf", "int", [1, 2, 4], 1),
            HyperparameterGrid("max_features", "categorical", ["sqrt", "log2", None], "sqrt"),
        ],
        complexity_score=5.0, interpretability_score=6.0,
        min_samples_recommended=1000, supports_feature_importance=True,
        notes="Robust ensemble method. Handles non-linearity and missing features well.",
    ),
    ModelConfig(
        model_id="xgb_clf", model_name="XGBoost Classifier",
        model_family="xgboost", library="xgboost",
        task_types=["classification"],
        hyperparameter_grids=[
            HyperparameterGrid("n_estimators", "int", [100, 200, 500], 100),
            HyperparameterGrid("max_depth", "int", [3, 5, 7, 9], 6),
            HyperparameterGrid("learning_rate", "float", [0.01, 0.05, 0.1, 0.3], 0.1),
            HyperparameterGrid("subsample", "float", [0.6, 0.8, 1.0], 1.0),
            HyperparameterGrid("colsample_bytree", "float", [0.6, 0.8, 1.0], 1.0),
            HyperparameterGrid("reg_alpha", "float", [0, 0.1, 1.0], 0),
            HyperparameterGrid("reg_lambda", "float", [1.0, 2.0, 5.0], 1.0),
        ],
        complexity_score=7.0, interpretability_score=5.0,
        min_samples_recommended=2000, supports_feature_importance=True,
        notes="State-of-the-art gradient boosting. Excellent for tabular data competitions.",
    ),
    ModelConfig(
        model_id="svm_clf", model_name="Support Vector Machine (RBF)",
        model_family="svm", library="sklearn",
        task_types=["classification"],
        hyperparameter_grids=[
            HyperparameterGrid("C", "float", [0.1, 1, 10, 100], 1.0),
            HyperparameterGrid("gamma", "categorical", ["scale", "auto", 0.001, 0.01], "scale"),
            HyperparameterGrid("kernel", "categorical", ["rbf", "poly", "sigmoid"], "rbf"),
        ],
        complexity_score=6.0, interpretability_score=3.0,
        min_samples_recommended=1000, supports_feature_importance=False,
        notes="Effective in high-dimensional spaces. Slow on large datasets.",
    ),
    ModelConfig(
        model_id="mlp_clf", model_name="Multi-Layer Perceptron",
        model_family="mlp", library="sklearn",
        task_types=["classification", "regression"],
        hyperparameter_grids=[
            HyperparameterGrid("hidden_layer_sizes", "categorical", [(64,), (128,), (64,32), (128,64,32)], (100,)),
            HyperparameterGrid("activation", "categorical", ["relu", "tanh", "logistic"], "relu"),
            HyperparameterGrid("learning_rate_init", "float", [0.001, 0.01, 0.1], 0.001),
            HyperparameterGrid("alpha", "float", [0.0001, 0.001, 0.01], 0.0001),
            HyperparameterGrid("max_iter", "int", [200, 500, 1000], 200),
        ],
        complexity_score=8.0, interpretability_score=2.0,
        min_samples_recommended=3000, supports_feature_importance=False,
        notes="Flexible neural network. Requires feature scaling.",
    ),
    ModelConfig(
        model_id="lr_reg", model_name="Linear Regression",
        model_family="linear_regression", library="sklearn",
        task_types=["regression"],
        hyperparameter_grids=[
            HyperparameterGrid("fit_intercept", "categorical", [True, False], True),
        ],
        complexity_score=1.0, interpretability_score=10.0,
        min_samples_recommended=100, supports_feature_importance=True,
        notes="Simplest baseline for regression. Assumes linear relationship.",
    ),
    ModelConfig(
        model_id="ridge_reg", model_name="Ridge Regression",
        model_family="ridge", library="sklearn",
        task_types=["regression"],
        hyperparameter_grids=[
            HyperparameterGrid("alpha", "float", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0),
            HyperparameterGrid("fit_intercept", "categorical", [True, False], True),
        ],
        complexity_score=2.0, interpretability_score=9.0,
        min_samples_recommended=200, supports_feature_importance=True,
        notes="L2 regularization. Handles multicollinearity well.",
    ),
    ModelConfig(
        model_id="gb_reg", model_name="Gradient Boosting Regressor",
        model_family="gradient_boosting", library="sklearn",
        task_types=["regression"],
        hyperparameter_grids=[
            HyperparameterGrid("n_estimators", "int", [100, 200, 300], 100),
            HyperparameterGrid("max_depth", "int", [3, 5, 7], 3),
            HyperparameterGrid("learning_rate", "float", [0.01, 0.05, 0.1], 0.1),
            HyperparameterGrid("subsample", "float", [0.6, 0.8, 1.0], 1.0),
        ],
        complexity_score=7.0, interpretability_score=4.0,
        min_samples_recommended=2000, supports_feature_importance=True,
        notes="Excellent performance on tabular regression. Prone to overfitting without regularization.",
    ),
    ModelConfig(
        model_id="kmeans_cl", model_name="K-Means Clustering",
        model_family="kmeans", library="sklearn",
        task_types=["clustering"],
        hyperparameter_grids=[
            HyperparameterGrid("n_clusters", "int", [2, 3, 4, 5, 6, 7, 8], 8),
            HyperparameterGrid("init", "categorical", ["k-means++", "random"], "k-means++"),
            HyperparameterGrid("n_init", "int", [10, 20, 30], 10),
            HyperparameterGrid("max_iter", "int", [100, 300, 500], 300),
        ],
        complexity_score=3.0, interpretability_score=7.0,
        min_samples_recommended=500, supports_feature_importance=False,
        notes="Simple, fast clustering. Assumes spherical clusters of equal size.",
    ),
    ModelConfig(
        model_id="dbscan_cl", model_name="DBSCAN",
        model_family="dbscan", library="sklearn",
        task_types=["clustering"],
        hyperparameter_grids=[
            HyperparameterGrid("eps", "float", [0.1, 0.3, 0.5, 1.0, 2.0], 0.5),
            HyperparameterGrid("min_samples", "int", [3, 5, 10, 15], 5),
            HyperparameterGrid("metric", "categorical", ["euclidean", "manhattan", "cosine"], "euclidean"),
        ],
        complexity_score=4.0, interpretability_score=5.0,
        min_samples_recommended=500, supports_feature_importance=False,
        notes="Discovers arbitrary-shaped clusters. Handles noise/outliers as separate class.",
    ),
]

class ModelSelector:
    """
    Selects and ranks ML models based on task type and experiment constraints.

    Usage
    -----
    selector = ModelSelector()
    result = selector.select(task_type="classification", objective="Detect credit card fraud")
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def select(self, task_type: str, objective: str = "") -> ModelSelectionResult:
        candidates = [m for m in MODEL_REGISTRY if task_type in m.task_types]
        if not candidates:
            candidates = [m for m in MODEL_REGISTRY if "classification" in m.task_types]

        ranked = self._rank_models(candidates, objective, task_type)
        primary = ranked[0]
        rationale = self._build_rationale(primary, task_type, objective)
        hp_summary = self._build_hyperparameter_summary(ranked[:3])

        result = ModelSelectionResult(
            task_type=task_type,
            objective=objective,
            ranked_models=ranked,
            primary_recommendation=primary,
            rationale=rationale,
            hyperparameter_summary=hp_summary,
        )
        self._print_summary(result)
        return result

    def _rank_models(self, candidates: List[ModelConfig], objective: str, task_type: str) -> List[ModelConfig]:
        lower = objective.lower()
        scores = {}
        for m in candidates:
            score = 5.0
            # Boost interpretable models if keywords present
            if any(x in lower for x in ["explain","interpret","understand","transparent"]):
                score += m.interpretability_score * 0.5
            # Boost simpler models for small datasets
            if any(x in lower for x in ["small","few","limited","sparse"]):
                score -= m.complexity_score * 0.3
            # Boost XGBoost/ensemble for high performance tasks
            if any(x in lower for x in ["best","highest","optimal","state-of-the-art"]):
                if m.model_family in ["xgboost","random_forest","gradient_boosting"]:
                    score += 3.0
            # Baseline penalty for very complex models without enough data
            if m.complexity_score > 7:
                score -= 1.0
            scores[m.model_id] = score + random.uniform(-0.1, 0.1)

        return sorted(candidates, key=lambda m: scores[m.model_id], reverse=True)

    def _build_rationale(self, model: ModelConfig, task_type: str, objective: str) -> str:
        return (
            f"'{model.model_name}' is recommended for this {task_type} task. "
            f"It offers a complexity score of {model.complexity_score}/10 and "
            f"interpretability of {model.interpretability_score}/10. "
            f"{model.notes}"
        )

    def _build_hyperparameter_summary(self, models: List[ModelConfig]) -> Dict[str, Dict[str, Any]]:
        summary = {}
        for m in models:
            summary[m.model_name] = {
                hp.param_name: {
                    "type": hp.param_type,
                    "search_space": hp.values,
                    "default": hp.default,
                }
                for hp in m.hyperparameter_grids
            }
        return summary

    def _print_summary(self, result: ModelSelectionResult) -> None:
        print(f"\nMODEL SELECTION | Task: {result.task_type}")
        print(f"Primary Recommendation: {result.primary_recommendation.model_name}")
        print(f"Rationale: {result.rationale}")
        print("Ranked Models:")
        for i, m in enumerate(result.ranked_models, 1):
            print(f"  {i}. {m.model_name} (complexity={m.complexity_score}, interpretability={m.interpretability_score})")

if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "classification"
    objective = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Predict customer churn"
    ModelSelector().select(task, objective)
