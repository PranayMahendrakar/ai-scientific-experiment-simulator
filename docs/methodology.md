# Experimental Methodology & Evaluation Framework

## Overview

This document describes the scientific methodology used by the **AI Scientific Experiment Simulator** to design, execute, and evaluate machine learning experiments from a free-text research objective.

---

## 1. Objective Parsing

The system uses **NLP keyword extraction** to parse the research objective:

1. The objective string is lowercased and tokenized.
2. A keyword bank maps terms to task types: `classification`, `regression`, `clustering`, `nlp`, `cv`.
3. The task type with the highest keyword match score is selected.
4. Fallback: defaults to `classification` if no keywords match.

**Example:**
```
Input:  "Classify medical images to detect pneumonia with high recall"
Output: task_type = "cv"  (keywords: "classify", "images", "detect")
```

---

## 2. Experiment Design

Each experiment plan contains **up to 5 variations**, selected from templates per task type:

| Task Type       | Variations                                                              |
|-----------------|-------------------------------------------------------------------------|
| classification  | Logistic Regression, Random Forest, XGBoost, SVM, MLP                 |
| regression      | Linear Regression, Ridge, Gradient Boosting, Lasso, SVR               |
| clustering      | K-Means, DBSCAN, Agglomerative, GMM, K-Means++                        |
| nlp / cv        | Task-specific deep learning configurations                             |

Each variation specifies:
- **Model family** (e.g., `random_forest`)
- **Tuning strategy**: `baseline` | `grid_search` | `random_search` | `bayesian`
- **Preprocessing pipeline**: e.g., `[standard_scaler, label_encoder, smote]`

---

## 3. Dataset Profiling

The system automatically estimates dataset characteristics:

- **Data type**: tabular, image, text, or time_series
- **Class balance**: balanced or imbalanced (detected via keywords like "fraud", "rare", "outlier")
- **Estimated samples**: 5,000 – 50,000 (synthetic simulation)
- **Estimated features**: varies by data type (tabular: 20, image: 50,176, text: 768)
- **Missing values**: randomly determined (50% probability)

---

## 4. Evaluation Framework

### 4.1 Cross-Validation Strategy

| Task Type       | Strategy            | Folds |
|-----------------|---------------------|-------|
| classification  | Stratified K-Fold   | 5     |
| regression      | K-Fold CV           | 5     |
| clustering      | Hold-out (no CV)    | 1     |
| cv / nlp        | Hold-out 80/20      | 1     |

### 4.2 Primary & Secondary Metrics

| Task Type       | Primary Metric          | Secondary Metrics                              |
|-----------------|-------------------------|------------------------------------------------|
| classification  | `f1_weighted`           | accuracy, precision_weighted, recall, roc_auc  |
| regression      | `rmse`                  | mae, r2_score, mape, explained_variance        |
| clustering      | `silhouette_score`      | davies_bouldin, calinski_harabasz, adj_rand    |
| nlp             | `f1_macro`              | accuracy, bleu, rouge_l, perplexity            |
| cv              | `mean_average_precision`| accuracy, iou, f1_macro, recall                |

### 4.3 Early Stopping

Training simulations implement **early stopping** with:
- Patience: 10 epochs
- Minimum delta: 0.001 (improvement threshold)
- Minimum epochs: 10 before triggering

---

## 5. Learning Curve Simulation

Each training fold simulates a **logarithmic convergence curve**:

```
train_metric = target × (1 - exp(-5 × epoch/total_epochs)) + Gaussian(0, 0.008)
val_metric   = target × (1 - exp(-4 × epoch/total_epochs)) × 0.97 + Gaussian(0, 0.012)
```

This produces realistic:
- Training-validation gap (overfitting effect at ~3%)
- Gaussian noise per epoch (measurement noise)
- Plateau behavior near target performance

---

## 6. Model Ranking

Models are ranked per experiment by:

1. **Primary metric** (descending for accuracy metrics, ascending for error metrics)
2. **Tie-breaking**: complexity score (simpler preferred)
3. **Context boosts**: interpretability boost for "explain" objectives; ensemble boost for "best/optimal" objectives

---

## 7. Result Aggregation

Cross-validation results are aggregated as:

```python
mean = sum(fold_metrics) / n_folds
std  = sqrt(sum((v - mean)^2 for v in fold_metrics) / (n_folds - 1))
```

Reported as: **metric = mean ± std** (e.g., `f1_weighted = 0.923 ± 0.008`)

---

## 8. Report Generation

The system generates two report formats:

### JSON Report
Machine-readable, contains full experiment metadata:
- Experiment plan, model configurations, hyperparameters
- Per-fold metrics, CV aggregations, learning curve data
- Ranked comparisons, insights, recommendations

### HTML Report (GitHub Pages)
Interactive dashboard featuring:
- Key metric cards
- Ranked comparison table
- Bar chart (model comparison)
- Radar chart (multi-metric)
- Learning curve visualization
- CV fold score breakdown
- AI-generated insights and recommendations

---

## 9. Insight Generation

The analyzer auto-generates insights based on:

- **Best model performance**: absolute score and interpretation
- **Performance spread**: range across variations, model family impact
- **Ensemble advantage**: comparison of boosting vs. linear models
- **Training efficiency**: fastest model identification
- **Early stopping**: overfitting signals

---

## 10. Recommendations

Automatic recommendations are generated based on:

1. Deploy the #1 ranked model (primary metric)
2. Bayesian hyperparameter optimization for top model
3. Feature importance analysis (for tree-based models)
4. Ensemble stacking if 3+ variations tested
5. Held-out test set validation before deployment

---

## References

- Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Breiman, L. (2001). Random Forests. Machine Learning.
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation.
