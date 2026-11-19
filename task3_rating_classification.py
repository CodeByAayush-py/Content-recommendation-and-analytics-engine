# %% [markdown]
# # Task 3 (Medium) – Netflix Audience Rating Classification
# The dataset's `rating` column is the **maturity rating** (TV-MA, PG-13 …), not a viewer score, so "audience rating
# category" is modelled as the audience group the rating belongs to: **Kids / Teens / Adults**.
# Ratings `NR` and `UR` (unrated, 82 rows) have no audience group and are excluded.

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from common import load_clean, genre_matrix, save_fig, RATING_GROUP

df = load_clean()

# %% [markdown]
# ## Step 1 – Analyze rating categories

# %%
print(df["rating"].value_counts().to_string())
d = df.dropna(subset=["audience"]).reset_index(drop=True)
print("\nAudience groups:\n", d["audience"].value_counts().to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
df["rating"].value_counts().plot.bar(ax=ax[0], color="#e50914"); ax[0].set_title("Maturity ratings")
pd.crosstab(d["audience"], d["type"], normalize="index").plot.bar(stacked=True, ax=ax[1], color=["#221f1f", "#e50914"])
ax[1].set_title("Movie / TV share by audience group"); ax[1].tick_params(axis="x", rotation=0)
save_fig("task3_rating_distribution.png")

# %% [markdown]
# ## Step 2 – Prepare training datasets
# Features: type, genres (multi-hot, ≥20 titles), top-15 primary countries, release year, year added,
# duration (minutes for movies / seasons for shows), director flag, number of genres.
# The target (`rating`) is never used as a feature.

# %%
G = genre_matrix(d, min_count=20)
top_c = d["primary_country"].value_counts().head(15).index
C = pd.get_dummies(d["primary_country"].where(d["primary_country"].isin(top_c), "Other"), prefix="c")
num = d[["release_year", "year_added", "month_added", "duration_value", "has_director", "n_genres"]].copy()
num = num.fillna(num.median())
num["is_tv"] = (d["type"] == "TV Show").astype(int)
X = pd.concat([G, C, num], axis=1).astype(float)
y = d["audience"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X.shape, Xtr.shape, Xte.shape)

# %% [markdown]
# ## Step 3 – Train classification models (baselines)

# %%
base = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
rows = []
for n, m in base.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append({"Model": n, "Stage": "baseline", "Accuracy": accuracy_score(yte, p), "Macro F1": f1_score(yte, p, average="macro")})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# %% [markdown]
# ## Step 4 – Optimize model performance (GridSearchCV, 5-fold, scoring = macro F1)
# Macro F1 is used because "Kids" is a minority class and accuracy alone would hide poor minority performance.

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=42)
grids = {
    "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                      {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
                      {"n_estimators": [200, 400], "max_depth": [10, 20, None], "min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"n_estimators": [150, 300], "max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]}),
}
tuned, cv_scores = {}, {}
for n, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
    tuned[n] = gs.best_estimator_; cv_scores[n] = gs.best_score_
    p = gs.best_estimator_.predict(Xte)
    print(f"{n}: best params {gs.best_params_}  CV macro-F1={gs.best_score_:.4f}")
    rows.append({"Model": n, "Stage": "tuned", "Accuracy": accuracy_score(yte, p), "Macro F1": f1_score(yte, p, average="macro")})

# %% [markdown]
# ## Step 5 – Evaluate prediction accuracy

# %%
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False)); res.to_csv("outputs/task3_model_comparison.csv", index=False)
maj = y.value_counts(normalize=True).max()
print(f"\nMajority-class baseline accuracy = {maj:.4f}")

piv = res.pivot(index="Model", columns="Stage", values="Macro F1")
piv.plot.barh(figsize=(8, 3.8), color=["#221f1f", "#e50914"]); plt.title("Macro F1 – baseline vs tuned"); plt.gca().invert_yaxis()
save_fig("task3_model_comparison.png")

best_name = max(cv_scores, key=cv_scores.get)  # pick by cross-validation on the training set, NOT by the test set
print("CV macro-F1 of tuned models:", {k: round(v, 4) for k, v in cv_scores.items()})
best = tuned[best_name]; pred = best.predict(Xte)
print(f"\nBest tuned model: {best_name}\n")
rep = classification_report(yte, pred, digits=3); print(rep)
open("outputs/task3_classification_report.txt", "w").write(f"Best tuned model: {best_name}\n\n{rep}")

ConfusionMatrixDisplay.from_predictions(yte, pred, labels=["Kids", "Teens", "Adults"], cmap="Reds")
plt.title(f"Confusion matrix – {best_name}")
save_fig("task3_confusion_matrix.png")

imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(15)
imp.plot.barh(color="#e50914", figsize=(7, 4.5), title=f"Top features – {best_name}")
save_fig("task3_feature_importance.png")

# %% [markdown]
# ## Bonus – full 12-class problem (original rating labels)
# Predicting the exact rating (e.g. TV-14 vs PG-13) is much harder because several labels are near-synonyms.

# %%
d12 = df[df["rating"].isin(RATING_GROUP)].reset_index(drop=True)
G12 = genre_matrix(d12, min_count=20)
C12 = pd.get_dummies(d12["primary_country"].where(d12["primary_country"].isin(top_c), "Other"), prefix="c")
n12 = d12[["release_year", "year_added", "month_added", "duration_value", "has_director", "n_genres"]].copy()
n12 = n12.fillna(n12.median()); n12["is_tv"] = (d12["type"] == "TV Show").astype(int)
X12 = pd.concat([G12, C12, n12], axis=1).astype(float); y12 = d12["rating"]
a, b, c_, e_ = train_test_split(X12, y12, test_size=0.2, stratify=y12, random_state=42)
rf12 = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(a, c_)
p12 = rf12.predict(b)
print(f"12-class Random Forest: accuracy={accuracy_score(e_, p12):.4f}, macro-F1={f1_score(e_, p12, average='macro'):.4f}, "
      f"majority baseline={y12.value_counts(normalize=True).max():.4f}")
