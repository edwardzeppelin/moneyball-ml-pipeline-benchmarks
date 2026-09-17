# Moneyball Machine Learning Regression & Benchmarking Framework

A comprehensive machine learning regression workspace using `scikit-learn` to analyze the OpenML **Moneyball** dataset (ID: 41021). This repository demonstrates end-to-end ML workflows including dataset preprocessing pipelines, cross-validated model benchmarking, hyperparameter grid optimization, and feature importance extraction.

---

## 📌 Features

* **Modular Pipelines**: Streamlined preprocessing using `ColumnTransformer` with numerical median imputation, scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`).
* **Model Benchmarking**: 5-fold cross-validated $R^2$ evaluation comparing linear models, kernel-based methods, and ensemble algorithms.
* **Hyperparameter Tuning**: Logarithmic parameter sweeps for alpha tuning in Ridge and Lasso, as well as joint $C$ and $\gamma$ grid searches for Support Vector Regression (SVR).
* **Feature Importance Analysis**: Unified post-processing that maps model coefficients and Gini/entropy importances back to original features after one-hot encoding.
* **Automated Visualizations**: Scatter matrices, performance vs. regularization curves, and comparative bar charts for feature impact.

---

## 🛠 Dataset & Preprocessing

The project utilizes the OpenML Moneyball dataset:
* **Target Variable**: Team performance metrics (`y`).
* **Categorical Features**: `Team`, `League`, `Playoffs`.
* **Numerical Features**: All remaining numeric statistics (e.g., runs scored, hits, errors).
* **Handling Missing Data**: Automated median imputation for numerical values.

---

## 🤖 Models Benchmarked

The framework compares performance with and without feature scaling across the following estimators:
* **Linear Models**: `LinearRegression`, `Ridge`, `Lasso`
* **Support Vector Machines**: `SVR` (RBF kernel)
* **Ensemble Methods**: `RandomForestRegressor`, `GradientBoostingRegressor`

---

## 📂 Project Structure

```text
.
├── main.py              # Main Python script with pipelines, model loops, and plotting
├── README.md            # Documentation
└── requirements.txt     # Python dependencies
```

## 💻 Installation & Usage
1. RequirementsEnsure Python 3.8+ is installed. Install the necessary dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn openml seaborn
```

2. ExecutionRun the main script to start data downloading, model evaluation, tuning, and plot rendering:

```bash
python main.py
```

## 📈 Analysis & Visualizations
1. Scatter Matrix: Explores numerical feature distributions and correlations.
2. Alpha Regularization Plot: Visualizes the impact of regularization strength ($\alpha \in [10^{-4}, 10^6]$) on $R^2$ scores for Ridge and Lasso regression[cite: 1].
3. SVM Parameter Heatmaps: Analyzes the trade-off between $C$ and $\gamma$ parameters for SVR with/without feature scaling.
4. Feature Importance Bar Charts: Ranks top features influencing model predictions across both linear models and tree-based ensembles[cite: 1].
