# experiment_planner.py
from __future__ import annotations
import uuid, random
from dataclasses import dataclass, field
from typing import List, Dict

TASK_KEYWORDS = {
    "classification": ["classify","classification","detect","categorize","label","diagnose","identify","spam","sentiment","fraud","churn","disease","cancer","tumor"],
    "regression": ["regress","forecast","estimate","price","sales","revenue","temperature","score","rating","continuous","amount","quantity"],
    "clustering": ["cluster","group","segment","partition","unsupervised","similar","anomaly","outlier"],
    "nlp": ["text","nlp","language","summarize","translation","named entity","ner","qa","document","corpus","bert","gpt"],
    "cv": ["image","vision","photo","pixel","cnn","convolutional","object detection","segmentation","face","ocr"],
}

@dataclass
class DatasetProfile:
    estimated_samples: int
    estimated_features: int
    class_balance: str
    missing_values: bool
    data_type: str

@dataclass
class ExperimentVariation:
    variation_id: str
    name: str
    description: str
    model_family: str
    tuning_strategy: str
    preprocessing: List[str]

@dataclass
class EvaluationStrategy:
    method: str
    n_splits: int
    test_size: float
    random_seed: int

@dataclass
class ExperimentPlan:
    plan_id: str
    objective: str
    task_type: str
    dataset_profile: DatasetProfile
    variations: List[ExperimentVariation]
    evaluation_strategy: EvaluationStrategy
    primary_metric: str
    secondary_metrics: List[str]
    notes: List[str] = field(default_factory=list)

class ExperimentPlanner:
    """Designs a full experiment plan from a research objective."""

    def __init__(self, max_variations=5, random_seed=42):
        self.max_variations = max_variations
        self.random_seed = random_seed
        random.seed(random_seed)

    def design(self, objective: str) -> ExperimentPlan:
        task_type = self._detect_task_type(objective)
        dataset_profile = self._estimate_dataset_profile(objective, task_type)
        eval_strategy = self._build_evaluation_strategy(task_type)
        variations = self._generate_variations(task_type)
        primary_metric, secondary_metrics = self._select_metrics(task_type)
        notes = self._generate_notes(objective, task_type)
        plan = ExperimentPlan(
            plan_id=str(uuid.uuid4())[:8], objective=objective, task_type=task_type,
            dataset_profile=dataset_profile, variations=variations[:self.max_variations],
            evaluation_strategy=eval_strategy, primary_metric=primary_metric,
            secondary_metrics=secondary_metrics, notes=notes,
        )
        self._print_summary(plan)
        return plan

    def _detect_task_type(self, objective: str) -> str:
        lower = objective.lower()
        scores = {t: sum(1 for kw in kws if kw in lower) for t, kws in TASK_KEYWORDS.items()}
        best = max(scores, key=lambda t: scores[t])
        return best if scores[best] > 0 else "classification"

    def _estimate_dataset_profile(self, objective, task_type):
        lower = objective.lower()
        data_type = "image" if task_type=="cv" else "text" if task_type=="nlp" else "time_series" if any(x in lower for x in ["time","series"]) else "tabular"
        class_balance = "imbalanced" if any(h in lower for h in ["fraud","rare","anomaly","outlier","minority"]) else "balanced"
        n_features = {"tabular":20,"image":50176,"text":768,"time_series":30}
        return DatasetProfile(random.randint(5000,50000), n_features.get(data_type,20), class_balance, random.random()>0.5, data_type)

    def _build_evaluation_strategy(self, task_type):
        if task_type in ("cv","nlp"):
            return EvaluationStrategy("hold_out", 1, 0.2, self.random_seed)
        return EvaluationStrategy("stratified_cv", 5, 0.2, self.random_seed)

    def _generate_variations(self, task_type):
        templates = {
            "classification": [
                ("Baseline Logistic Regression","logistic_regression","baseline",["standard_scaler","label_encoder"]),
                ("Random Forest (Grid Search)","random_forest","grid_search",["standard_scaler","label_encoder"]),
                ("XGBoost (Random Search)","xgboost","random_search",["standard_scaler","smote"]),
                ("SVM with RBF Kernel","svm","grid_search",["standard_scaler","pca"]),
                ("MLP Neural Network","mlp","random_search",["standard_scaler"]),
            ],
            "regression": [
                ("Baseline Linear Regression","linear_regression","baseline",["standard_scaler"]),
                ("Ridge Regression","ridge","grid_search",["standard_scaler"]),
                ("Gradient Boosting","gradient_boosting","random_search",["standard_scaler","power_transformer"]),
                ("Lasso Regression","lasso","grid_search",["standard_scaler","feature_selection"]),
                ("SVR with RBF Kernel","svr","grid_search",["standard_scaler","pca"]),
            ],
            "clustering": [
                ("K-Means Baseline","kmeans","baseline",["standard_scaler"]),
                ("DBSCAN","dbscan","grid_search",["standard_scaler","pca"]),
                ("Agglomerative Clustering","agglomerative","baseline",["standard_scaler"]),
                ("Gaussian Mixture Model","gmm","random_search",["standard_scaler","pca"]),
                ("K-Means++ Optimized","kmeans_plus","bayesian",["standard_scaler","umap"]),
            ],
        }
        items = templates.get(task_type, templates["classification"])
        return [ExperimentVariation(f"V{i+1:02d}",name,f"Variation using {fam} with {strat}",fam,strat,prep) for i,(name,fam,strat,prep) in enumerate(items)]

    def _select_metrics(self, task_type):
        m = {"classification":("f1_weighted",["accuracy","precision_weighted","recall_weighted","roc_auc"]),
             "regression":("rmse",["mae","r2_score","mape","explained_variance"]),
             "clustering":("silhouette_score",["davies_bouldin","calinski_harabasz","adjusted_rand"]),
             "nlp":("f1_macro",["accuracy","bleu","rouge_l","perplexity"]),
             "cv":("mean_average_precision",["accuracy","iou","f1_macro","recall"])}
        return m.get(task_type, m["classification"])

    def _generate_notes(self, objective, task_type):
        notes = [f"Objective parsed as a {task_type} problem.",
                 "Ensure preprocessing is applied consistently across all splits.",
                 "Use stratified splits if class imbalance is detected."]
        if any(x in objective.lower() for x in ["imbalanced","fraud"]):
            notes.append("Consider SMOTE oversampling for class imbalance.")
        return notes

    def _print_summary(self, plan):
        print(f"EXPERIMENT PLAN [{plan.plan_id}] | Task: {plan.task_type} | Variations: {len(plan.variations)}")
        for v in plan.variations:
            print(f"  [{v.variation_id}] {v.name}")

if __name__ == "__main__":
    import sys
    objective = " ".join(sys.argv[1:]) or "Classify emails as spam or not spam"
    ExperimentPlanner().design(objective)
