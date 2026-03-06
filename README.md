# 🧪 AI Scientific Experiment Simulator

An AI-driven framework that **automatically designs, simulates, and analyzes machine learning experiments**. Given a research objective, it generates experiment designs, selects models, configures hyperparameters, runs simulated training pipelines, and produces comprehensive evaluation reports.

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://pranaymahendr akar.github.io/ai-scientific-experiment-simulator/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Purpose

Automatically design and simulate machine learning experiments from a high-level research objective — eliminating manual trial-and-error in ML research workflows.

---

## 🏗️ Architecture

```
ai-scientific-experiment-simulator/
├── modules/
│   ├── experiment_planner.py     # Designs experiment variations
│   ├── model_selector.py         # Selects optimal ML models
│   ├── training_pipeline.py      # Simulates training & evaluation
│   └── result_analyzer.py        # Analyzes & compares results
├── simulator.py                  # Main orchestrator
├── config.yaml                   # Default configuration
├── docs/
│   └── methodology.md            # Experimental methodology
├── index.html                    # GitHub Pages dashboard
└── README.md
```

---

## 📥 Input

A **research objective** string, e.g.:

```
"Classify medical images to detect pneumonia with high recall"
```

---

## 📤 Output

| Output | Description |
|--------|-------------|
| **Experiment Design** | Task type, dataset profile, evaluation strategy |
| **Model Configuration** | Selected algorithms with rationale |
| **Hyperparameters** | Optimized parameter grids per model |
| **Evaluation Metrics** | Accuracy, F1, AUC-ROC, Precision, Recall |

---

## 🔧 Modules

### 1. `experiment_planner`
- Parses research objective using NLP keyword extraction
- - Determines task type (classification, regression, clustering, etc.)
  - - Designs multiple experiment variations (baseline + tuned + ensembles)
    - - Outputs structured experiment blueprint
     
      - ### 2. `model_selector`
      - - Maps task type → candidate model families
        - - Scores models by complexity, interpretability, data requirements
          - - Returns ranked model list with configuration templates
           
            - ### 3. `training_pipeline`
            - - Simulates training with synthetic performance metrics
              - - Applies cross-validation strategy
                - - Tracks per-epoch metrics (loss, accuracy, AUC)
                  - - Supports early stopping simulation
                   
                    - ### 4. `result_analyzer`
                    - - Aggregates results across all experiment variations
                      - - Generates comparison tables and ranking
                        - - Identifies best-performing configuration
                          - - Produces HTML/JSON reports for GitHub Pages
                           
                            - ---

                            ## 🚀 Quick Start

                            ```bash
                            # Clone the repository
                            git clone https://github.com/PranayMahendrakar/ai-scientific-experiment-simulator.git
                            cd ai-scientific-experiment-simulator

                            # Install dependencies
                            pip install -r requirements.txt

                            # Run the simulator
                            python simulator.py --objective "Predict customer churn with high precision"
                            ```

                            ---

                            ## 📊 Experiment Report (Sample)

                            ```
                            Research Objective : Predict customer churn with high precision
                            Task Type          : Binary Classification
                            Variations Designed: 5

                            ┌─────────────────────────┬──────────┬────────┬───────┬────────┐
                            │ Model                   │ Accuracy │    F1  │  AUC  │  Rank  │
                            ├─────────────────────────┼──────────┼────────┼───────┼────────┤
                            │ Random Forest (tuned)   │  0.923   │ 0.918  │ 0.961 │   1    │
                            │ XGBoost (baseline)      │  0.911   │ 0.905  │ 0.954 │   2    │
                            │ Logistic Regression     │  0.874   │ 0.869  │ 0.921 │   3    │
                            │ SVM (RBF kernel)        │  0.887   │ 0.881  │ 0.933 │   4    │
                            │ Neural Network (MLP)    │  0.901   │ 0.896  │ 0.947 │   5    │
                            └─────────────────────────┴──────────┴────────┴───────┴────────┘

                            Best Model: Random Forest (tuned)
                            Recommended Hyperparameters: n_estimators=200, max_depth=12, min_samples_split=5
                            ```

                            ---

                            ## 📖 Documentation

                            - [Experimental Methodology](docs/methodology.md)
                            - - [Evaluation Framework](docs/methodology.md#evaluation-framework)
                              - - [Live Dashboard](https://pranaymahendr akar.github.io/ai-scientific-experiment-simulator/)
                               
                                - ---

                                ## 🛠️ Configuration

                                Edit `config.yaml` to customize default settings:

                                ```yaml
                                experiment:
                                  max_variations: 5
                                  cv_folds: 5
                                  random_seed: 42

                                models:
                                  classification: [random_forest, xgboost, logistic_regression, svm, mlp]
                                  regression: [linear_regression, ridge, lasso, gradient_boosting, svr]

                                evaluation:
                                  primary_metric: f1_weighted
                                  secondary_metrics: [accuracy, auc_roc, precision, recall]
                                ```

                                ---

                                ## 📄 License

                                MIT License — see [LICENSE](LICENSE) for details.
