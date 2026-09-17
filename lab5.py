import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml as oml
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import time

moneyball = oml.datasets.get_dataset(41021)
X, y, _, attribute_names = moneyball.get_data(target = moneyball.default_target_attribute)

pd.isnull(X).any()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_clean_array = imputer.fit_transform(X[attribute_names[2:]])
X_clean = pd.DataFrame(X_clean_array, columns = attribute_names[2:])

from pandas.plotting import scatter_matrix

copyframe = X_clean.copy()
copyframe['y'] = pd.Series(y, index=copyframe.index)
scatter_matrix(copyframe, c=y, figsize=(25,25), marker='o', s=20, alpha=.8, cmap='viridis')

plt.tight_layout()
#plt.show()

#Exercise 1: Build a pipeline

def build_pipeline(regressor, numerical, categorical, scaling=False):
    
    numerical_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]

    if scaling:
        numerical_steps.append(('scaler', StandardScaler()))
        
    numerical_pipeline = Pipeline(numerical_steps)
    
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical),
        ('cat', categorical_pipeline, categorical)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    return pipeline

regressor = LinearRegression()

categorical = ['Team', 'League', 'Playoffs']
numerical = [col for col in X.columns if col not in categorical]

pipeline = build_pipeline(
    regressor=regressor,
    numerical=numerical,
    categorical=categorical,
    scaling=False
)

scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

print(f"Cross-validation R2 scores: {scores}")
print(f"Mean R2 score: {scores.mean():.4f}")

#Exercise 3: First Benchmark

models = [
    ('LinearRegression', LinearRegression()),
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('SVM', SVR(kernel='rbf')),
    ('RandomForest', RandomForestRegressor()),
    ('GradientBoosting', GradientBoostingRegressor())
]

print(f"{'Model':<20} {'Scaling':<10} {'Mean R2':>10}")
print("-" * 45)

for name, regressor in models:
    for scaling in [False, True]:
        pipeline = build_pipeline(
            regressor=regressor,
            numerical=numerical,
            categorical=categorical,
            scaling=scaling
        )
        
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        
        scale_label = "Scaled" if scaling else "Non-Scaled"
        print(f"{name:<20} {scale_label:<10} {scores.mean():>10.4f}")


#Exercise 4: Tuning linear models

alphas = np.logspace(-4, 6, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge_pipe = build_pipeline(
        regressor=Ridge(alpha=alpha),
        numerical=numerical,
        categorical=categorical,
        scaling=True
    )
    scores = cross_val_score(ridge_pipe, X, y, cv=5, scoring='r2')
    ridge_scores.append(scores.mean())
    
    lasso_pipe = build_pipeline(
        regressor=Lasso(alpha=alpha, max_iter=10000),
        numerical=numerical,
        categorical=categorical,
        scaling=True
    )
    scores = cross_val_score(lasso_pipe, X, y, cv=5, scoring='r2')
    lasso_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, label='Ridge', marker='o')
plt.plot(alphas, lasso_scores, label='Lasso', marker='s')
plt.xscale('log')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Mean R² Score (5-fold CV)')
plt.title('Effect of Alpha on Ridge and Lasso Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5, label='R² = 0')
plt.tight_layout()
#plt.show()

best_ridge_idx = np.argmax(ridge_scores)
best_lasso_idx = np.argmax(lasso_scores)
print(f"Best Ridge:  alpha={alphas[best_ridge_idx]:.4f}, R²={ridge_scores[best_ridge_idx]:.4f}")
print(f"Best Lasso:  alpha={alphas[best_lasso_idx]:.4f}, R²={lasso_scores[best_lasso_idx]:.4f}")


# Exercise 5: Tuning SVMs

'''

def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    values_flat = values.flatten()
    for i, (p, color) in enumerate(zip(img.get_paths(), img.get_facecolors())):
        x, y = p.vertices[:-2, :].mean(0)
        value = values_flat[i]
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img

C_values = np.logspace(-6, 6, 13)
gamma_values = np.logspace(-6, 6, 13)

results = np.zeros((len(gamma_values), len(C_values)))

print("Tuning SVM (without scaling)...")
for i, gamma in enumerate(gamma_values):
    for j, C in enumerate(C_values):
        pipeline = build_pipeline(
            regressor=SVR(kernel='rbf', C=C, gamma=gamma),
            numerical=numerical,
            categorical=categorical,
            scaling=False
        )
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='r2')
        results[i, j] = scores.mean()

plt.figure(figsize=(12, 10))
heatmap(results,
        xlabel='C', 
        ylabel='gamma', 
        xticklabels=[f'{c:.0e}' for c in C_values], 
        yticklabels=[f'{g:.0e}' for g in gamma_values],
        cmap='viridis',
        fmt="%0.2f")
plt.title('SVM R² Score (without scaling)')
plt.tight_layout()
#plt.show()

results_scaled = np.zeros((len(gamma_values), len(C_values)))

print("Tuning SVM (with scaling)...")
for i, gamma in enumerate(gamma_values):
    for j, C in enumerate(C_values):
        pipeline = build_pipeline(
            regressor=SVR(kernel='rbf', C=C, gamma=gamma),
            numerical=numerical,
            categorical=categorical,
            scaling=True
        )
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='r2')
        results_scaled[i, j] = scores.mean()

plt.figure(figsize=(12, 10))
heatmap(results_scaled,
        xlabel='C', 
        ylabel='gamma', 
        xticklabels=[f'{c:.0e}' for c in C_values], 
        yticklabels=[f'{g:.0e}' for g in gamma_values],
        cmap='viridis',
        fmt="%0.2f")
plt.title('SVM R² Score (with scaling)')
plt.tight_layout()
#plt.show()

best_idx = np.unravel_index(np.argmax(results), results.shape)
best_idx_scaled = np.unravel_index(np.argmax(results_scaled), results_scaled.shape)

print(f"\nBest without scaling: C={C_values[best_idx[1]]:.0e}, gamma={gamma_values[best_idx[0]]:.0e}, R²={results[best_idx]:.4f}")
print(f"Best with scaling:    C={C_values[best_idx_scaled[1]]:.0e}, gamma={gamma_values[best_idx_scaled[0]]:.0e}, R²={results_scaled[best_idx_scaled]:.4f}")
'''

# Exercise 6: Feature importance
results = {}

for name, model in [('Ridge', Ridge(alpha=10)), ('Lasso', Lasso(alpha=1, max_iter=10000))]:
    pipe = build_pipeline(model, numerical, categorical, scaling=True)
    pipe.fit(X, y)
    
    coefs = np.abs(pipe.named_steps['regressor'].coef_)
    
    imp = {}
    for feat, val in zip(numerical, coefs[:len(numerical)]):
        imp[feat] = val

    for feat, val in zip(pipe.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['encoder']
                         .get_feature_names_out(categorical),
                         coefs[len(numerical):]):

        orig = feat.split('_')[0] if '_' in feat else feat
        imp[orig] = imp.get(orig, 0) + val
    results[name] = pd.Series(imp).sort_values(ascending=False)
    print(f"{name} top 5:\n{results[name].head()}\n")

for name, model in [('RandomForest', RandomForestRegressor(random_state=42)), 
                    ('GradientBoosting', GradientBoostingRegressor(random_state=42))]:
    pipe = build_pipeline(model, numerical, categorical, scaling=True)
    pipe.fit(X, y)
    imp_vals = pipe.named_steps['regressor'].feature_importances_
    imp = {}
    for feat, val in zip(numerical, imp_vals[:len(numerical)]):
        imp[feat] = val
    for feat, val in zip(pipe.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['encoder']
                         .get_feature_names_out(categorical),
                         imp_vals[len(numerical):]):
        orig = feat.split('_')[0] if '_' in feat else feat
        imp[orig] = imp.get(orig, 0) + val
    results[name] = pd.Series(imp).sort_values(ascending=False)
    print(f"{name} top 5:\n{results[name].head()}\n")

plt.figure(figsize=(10, 6))
pd.DataFrame({'Ridge': results['Ridge'], 'Lasso': results['Lasso']}).iloc[:15].plot(kind='barh')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance: Linear Models')
plt.gca().invert_yaxis()
plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 6))
pd.DataFrame({'RandomForest': results['RandomForest'], 'GradientBoosting': results['GradientBoosting']}).iloc[:15].plot(kind='barh')
plt.xlabel('Feature Importance (0-1 scale)')
plt.title('Feature Importance: Tree-Based Models')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop-5 features per model")
for name in results:
    top5 = results[name].head(5).index.tolist()
    print(f"{name:15}: {top5}")