# %% [markdown]
# # Task 2 (Easy) – Content Type Prediction (Movie vs TV Show)
# Binary classification. **Important:** two columns literally contain the answer, so using them is *data leakage*:
# * `duration` ("90 min" vs "2 Seasons")
# * `listed_in` genre labels ("Dramas" only appears on movies, "TV Dramas" only on shows)
# We demonstrate both, then build an honest model from type-neutral features.

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from common import load_clean, genre_matrix, save_fig

df = load_clean()
df["y"] = (df["type"] == "TV Show").astype(int)
print(df["type"].value_counts(normalize=True).round(3))

# %% [markdown]
# ## Step 1 – Select relevant features
# * Leakage check: a one-feature rule on `duration_unit` is 100% accurate.

# %%
leak = (df["duration_unit"].eq("min")) == (df["type"] == "Movie")
print(f"'duration' alone predicts type with accuracy = {leak.mean():.3f}  -> leakage, excluded from the model")
e = df[["genres", "type"]].explode("genres").reset_index(drop=True)
purity = pd.crosstab(e["genres"], e["type"])
print(f"{(purity.min(axis=1) == 0).sum()} of {len(purity)} raw genre labels appear in only ONE type -> genre labels leak too")

# %% [markdown]
# ## Step 2 – Encode categorical variables
# Genres are first folded into type-neutral themes (Drama, Comedy, Horror…) → multi-hot, rating → one-hot, country (top 15, rest = Other) → one-hot, plus numeric year features.

# %%
G = genre_matrix(df, min_count=20, col="themes").drop(columns=["g_Other"], errors="ignore")  # "Other" = bare Movies/TV Shows label
top_c = df["primary_country"].value_counts().head(15).index
country = pd.get_dummies(df["primary_country"].where(df["primary_country"].isin(top_c), "Other"), prefix="c")
rating = pd.get_dummies(df["rating"], prefix="r")
num = df[["release_year", "year_added", "month_added", "has_director", "n_genres"]].copy()
num["year_added"] = num["year_added"].fillna(num["year_added"].median())
num["month_added"] = num["month_added"].fillna(num["month_added"].median())
num["lag_years"] = num["year_added"] - num["release_year"]

X = pd.concat([G, country, rating, num], axis=1).astype(float)
y = df["y"]
print("Feature matrix:", X.shape)

# %% [markdown]
# ## Step 3 – Train classification models (stratified 80/20 split)

# %%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
rows, fitted = [], {}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); fitted[name] = m
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(yte, p), "Precision": precision_score(yte, p),
                 "Recall": recall_score(yte, p), "F1": f1_score(yte, p),
                 "CV acc (5-fold)": cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean()})

# %% [markdown]
# ## Step 4 & 5 – Evaluate performance and compare accuracy

# %%
res = pd.DataFrame(rows).set_index("Model").round(4).sort_values("F1", ascending=False)
baseline = max(y.mean(), 1 - y.mean())
print(res.to_string()); print(f"\nMajority-class baseline accuracy = {baseline:.4f}")
res.to_csv("outputs/task2_model_comparison.csv")

fig, ax = plt.subplots(figsize=(8, 3.8))
res[["Accuracy", "F1"]].plot.barh(ax=ax, color=["#e50914", "#221f1f"])
ax.axvline(baseline, ls="--", c="grey"); ax.text(baseline + .005, 4.45, "majority-class baseline", color="grey", fontsize=8)
ax.set_xlim(0.5, 1); ax.invert_yaxis(); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False); ax.set_title("Movie vs TV Show – model comparison (no duration feature)")
save_fig("task2_model_comparison.png")

best = res.index[0]
print("Best model:", best)
ConfusionMatrixDisplay.from_estimator(fitted[best], Xte, yte, display_labels=["Movie", "TV Show"], cmap="Reds")
plt.title(f"Confusion matrix – {best}")
save_fig("task2_confusion_matrix.png")

rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(12)
imp.plot.barh(color="#e50914", figsize=(7, 4), title="Top features – Random Forest")
save_fig("task2_feature_importance.png")

# **Reading the result:** `has_director` is the strongest honest signal – in this dataset 97% of movies list a director
# but only 9% of TV shows do. That is a genuine pattern (and a data-collection artefact), so it is kept but disclosed.

# %% [markdown]
# ## For comparison: leaky models – shows why those columns must be excluded

# %%
def rf_acc(Xm):
    a, b, c, d = train_test_split(Xm, y, test_size=0.2, stratify=y, random_state=42)
    return accuracy_score(d, RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(a, c).predict(b))

X_raw_genres = pd.concat([genre_matrix(df, min_count=20), country, rating, num], axis=1).astype(float)
Xl = X.copy(); Xl["is_min"] = (df["duration_unit"] == "min").astype(int)
leak_tbl = pd.Series({"Honest (type-neutral themes, no duration)": rf_acc(X),
                      "Honest, but without has_director flag": rf_acc(X.drop(columns=["has_director"])),
                      "Leaky: raw genre labels": rf_acc(X_raw_genres),
                      "Leaky: + duration": rf_acc(Xl)}).round(4)
print(leak_tbl.to_string()); leak_tbl.to_csv("outputs/task2_leakage_demo.csv")
