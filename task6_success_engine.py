# %% [markdown]
# # Task 6 (Advanced) – Netflix Content Success Analytics Engine
# End-to-end pipeline: **features → multiple models → comparison → automated insights → visual report → saved model.**
#
# **Defining "success".** The dataset has no viewership, popularity or review data, so success must be a proxy.
# We use **TV show renewal: a show that ran for 2+ seasons** (a real business signal – Netflix only renews shows that
# perform). Movies have no equivalent signal, so they are covered by the descriptive catalogue analysis only.
# Limits of this proxy are listed at the end.

# %%
import os, base64, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, RocCurveDisplay
from common import load_clean, save_fig, OUT

df = load_clean()
tv = df[df["type"] == "TV Show"].copy()
tv["success"] = (tv["duration_value"] >= 2).astype(int)
print(f"{len(tv)} TV shows; success rate (2+ seasons) = {tv['success'].mean():.1%}")

# %% [markdown]
# ## Step 1 – Advanced feature engineering
# Every feature must be knowable *without* using the number of seasons (the target):
# | feature | idea |
# |---|---|
# | theme multi-hot, `n_themes` | genre mix and breadth |
# | `primary_country`, `n_countries`, `is_coproduction` | origin and international co-production |
# | `director_known`, `director_freq` | prolific/credited directors (frequency from the whole catalogue; no target used) |
# | `audience` | Kids / Teens / Adults from the maturity rating |
# | `release_year`, `year_added`, `lag_years`, `fast_add`, `month_added` | timing; `fast_add` = added within 1 year of release (typical of Netflix originals) |
# | `title_words`, `has_colon`, `has_number` | franchise / spin-off / sequel hints in the title |
# `duration` is **excluded** – it is the target.

# %%
dir_counts = df["director"].dropna().str.split(",").explode().str.strip().value_counts()
def dir_freq(s):
    return 0 if pd.isna(s) else max(dir_counts.get(x.strip(), 0) for x in s.split(","))

tv["n_themes"] = tv["themes"].str.len()
tv["n_countries"] = tv["country"].fillna("").apply(lambda s: len([c for c in s.split(",") if c.strip()]))
tv["is_coproduction"] = (tv["n_countries"] > 1).astype(int)
tv["director_known"] = tv["has_director"]
tv["director_freq"] = tv["director"].apply(dir_freq)
tv["lag_years"] = (tv["year_added"] - tv["release_year"]).clip(lower=-1, upper=15)
tv["fast_add"] = (tv["lag_years"] <= 1).astype(int)
tv["title_words"] = tv["title"].str.split().str.len()
tv["has_colon"] = tv["title"].str.contains(":").astype(int)
tv["has_number"] = tv["title"].str.contains(r"\d").astype(int)
tv["year_added"] = tv["year_added"].fillna(tv["year_added"].median())
tv["month_added"] = tv["month_added"].fillna(6)

theme_counts = tv["themes"].explode().value_counts()
themes_kept = [t for t in theme_counts.index if theme_counts[t] >= 30 and t != "Other"]
for t in themes_kept:
    tv["th_" + t] = tv["themes"].apply(lambda xs, t=t: int(t in xs))
top_c = tv["primary_country"].value_counts().head(12).index
tv["country_grp"] = tv["primary_country"].where(tv["primary_country"].isin(top_c), "Other")

NUM = ["release_year", "year_added", "lag_years", "month_added", "n_themes", "n_countries", "director_freq", "title_words"]
BIN = ["is_coproduction", "director_known", "fast_add", "has_colon", "has_number"] + ["th_" + t for t in themes_kept]
CAT = ["country_grp", "audience"]
FEATS = NUM + BIN + CAT
X, y = tv[FEATS], tv["success"]
print(f"{len(FEATS)} raw features ({len(themes_kept)} theme flags)")

# %% [markdown]
# ## Step 2 – Build multiple ML models (one reusable `Pipeline` each)
# Preprocessing lives *inside* the pipeline so cross-validation never leaks test information, and the whole thing
# can be saved and reused in production (the MLOps part).

# %%
def make_pre(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("bin", "passthrough", BIN),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

models = {
    "Baseline (majority)": Pipeline([("pre", make_pre(False)), ("m", DummyClassifier(strategy="prior"))]),
    "Logistic Regression": Pipeline([("pre", make_pre(True)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))]),
    "Random Forest": Pipeline([("pre", make_pre(False)), ("m", RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42))]),
    "Gradient Boosting": Pipeline([("pre", make_pre(False)), ("m", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42))]),
    "HistGradientBoosting": Pipeline([("pre", make_pre(False)), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))]),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# %% [markdown]
# ## Step 3 – Compare model performance
# Primary metric is **ROC-AUC** (threshold-free; 0.5 = coin flip). We also report PR-AUC, F1 and accuracy on the
# untouched 20% test set, and 5-fold CV AUC on the training set.

# %%
rows, fitted = [], {}
for n, p in models.items():
    cv_auc = cross_val_score(p, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
    p.fit(Xtr, ytr); fitted[n] = p
    prob = p.predict_proba(Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
    rows.append({"Model": n, "CV ROC-AUC": cv_auc, "Test ROC-AUC": roc_auc_score(yte, prob),
                 "Test PR-AUC": average_precision_score(yte, prob), "Test F1": f1_score(yte, pred), "Test Acc": accuracy_score(yte, pred)})
res = pd.DataFrame(rows).set_index("Model").round(3)
print(res.to_string()); res.to_csv(f"{OUT}/task6_model_comparison.csv")

# %% [markdown]
# ### Model optimisation – small grid search on the best model family (by CV AUC)

# %%
best_family = res.drop("Baseline (majority)")["CV ROC-AUC"].idxmax()
print("Best family by CV:", best_family)
grids = {
    "Gradient Boosting": {"m__n_estimators": [100, 200, 300], "m__max_depth": [2, 3], "m__learning_rate": [0.03, 0.05, 0.1]},
    "HistGradientBoosting": {"m__max_depth": [2, 3, 4], "m__learning_rate": [0.03, 0.05, 0.1], "m__max_iter": [100, 200]},
    "Random Forest": {"m__min_samples_leaf": [2, 3, 6], "m__max_depth": [None, 8, 12]},
    "Logistic Regression": {"m__C": [0.05, 0.1, 0.3, 1.0]},
}
gs = GridSearchCV(models[best_family], grids[best_family], cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
final = gs.best_estimator_
prob = final.predict_proba(Xte)[:, 1]
print(f"Tuned {best_family}: params={gs.best_params_}  CV AUC={gs.best_score_:.3f}  TEST AUC={roc_auc_score(yte, prob):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
res["Test ROC-AUC"].sort_values().plot.barh(ax=ax[0], color="#e50914"); ax[0].set_xlim(0.4, 0.9); ax[0].set_title("Test ROC-AUC by model")
for n in ["Baseline (majority)", "Logistic Regression", best_family]:
    RocCurveDisplay.from_estimator(final if n == best_family else fitted[n], Xte, yte, ax=ax[1], name=n + (" (tuned)" if n == best_family else ""))
ax[1].set_title("ROC curves"); ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
save_fig("task6_model_comparison.png")

# %% [markdown]
# ## Step 4 – Generate automated insights
# Everything below is computed from the data and written into sentences by code, so re-running on a new dataset
# regenerates the report automatically. First: what drives the model (permutation importance on the test set).

# %%
pi = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
imp_pretty = imp.rename(lambda c: "theme: " + c[3:] if c.startswith("th_") else c)
print(imp_pretty.head(10).round(4).to_string())

# renewal rate by segment (with minimum support), lift vs base rate
base = tv["success"].mean()
def seg_table(col, min_n=40, explode=False):
    d = tv[[col, "success"]].explode(col) if explode else tv[[col, "success"]]
    g = d.groupby(col)["success"].agg(["mean", "size"]); g = g[g["size"] >= min_n]
    g["lift"] = g["mean"] / base
    return g.sort_values("mean", ascending=False)
by_theme = seg_table("themes", explode=True).drop(index="Other", errors="ignore")
by_country = seg_table("primary_country")
by_aud = seg_table("audience", 20)
by_year = tv[tv["year_added"] >= 2016].groupby("year_added")["success"].mean()

# out-of-fold scores -> renewal candidates among 1-season shows (never scored by a model that saw them)
tv["p_success"] = cross_val_predict(final, X, y, cv=cv, method="predict_proba")[:, 1]
cands = tv[tv["duration_value"] == 1].sort_values("p_success", ascending=False).head(10)[["title", "release_year", "primary_country", "audience", "p_success"]]
cands["p_success"] = cands["p_success"].round(3)
oof_auc = roc_auc_score(y, tv["p_success"])

# catalogue-wide patterns (movies + TV)
yearly = df[df["year_added"] >= 2016].groupby(["year_added", "type"]).size().unstack()
peak_year = int(yearly.sum(axis=1).idxmax())
tv_share = yearly["TV Show"] / yearly.sum(axis=1)
top_country = df["primary_country"].value_counts()
top_country = top_country[top_country.index != "Unknown"]

insights = []
insights.append(f"**Catalogue growth:** titles added per year peaked in {peak_year} ({int(yearly.sum(axis=1).max()):,} titles). "
                f"TV's share of additions moved from {tv_share.loc[2016]:.0%} (2016) to {tv_share.loc[2018]:.0%} (2018) to {tv_share.loc[2021]:.0%} (2021).")
insights.append(f"**Where content comes from:** {top_country.index[0]} ({top_country.iloc[0]/len(df):.0%} of titles), then {top_country.index[1]} "
                f"({top_country.iloc[1]/len(df):.0%}) and {top_country.index[2]} ({top_country.iloc[2]/len(df):.0%}).")
insights.append(f"**Base rate:** {base:.0%} of TV shows in the catalogue ran for 2+ seasons.")
t1, t2 = by_theme.index[0], by_theme.index[1]
b1, b2 = by_theme.index[-1], by_theme.index[-2]
insights.append(f"**Genres that get renewed:** {t1} ({by_theme.loc[t1,'mean']:.0%}, n={int(by_theme.loc[t1,'size'])}) and {t2} "
                f"({by_theme.loc[t2,'mean']:.0%}) lead; {b1} ({by_theme.loc[b1,'mean']:.0%}) and {b2} ({by_theme.loc[b2,'mean']:.0%}) lag.")
c1, cl = by_country.index[0], by_country.index[-1]
insights.append(f"**Countries (≥40 shows):** best renewal rate is {c1} ({by_country.loc[c1,'mean']:.0%}), lowest is {cl} ({by_country.loc[cl,'mean']:.0%}).")
insights.append("**Audience:** " + ", ".join(f"{a} {r['mean']:.0%} (n={int(r['size'])})" for a, r in by_aud.iterrows()) + " renewal rate.")
insights.append(f"**Model:** tuned {best_family} separates renewed from one-season shows with test ROC-AUC {roc_auc_score(yte, prob):.2f} "
                f"(0.50 = chance; out-of-fold on all shows {oof_auc:.2f}). Top drivers: " + ", ".join(imp_pretty.index[:4]) + ".")
insights.append(f"**Timing:** the renewal rate of shows added in 2021 is {by_year.loc[2021]:.0%} vs {by_year.loc[2016]:.0%} for 2016 additions. "
                f"The data cannot say why (catalogue mix, licensing and sampling all change by year), so the model only uses it as a weak signal.")
for i in insights: print("-", i)
print("\nRenewal-profile matches (single-season shows scoring most like renewed shows; NOT real renewal predictions):"); print(cands.to_string(index=False))

# %% [markdown]
# ## Step 5 – Present findings through visual reports
# One dashboard image + a self-contained HTML report (`outputs/task6_report.html`) built entirely by code.

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
fig.suptitle("Netflix Content Success Analytics – TV renewal (2+ seasons)", fontsize=15, fontweight="bold")
yearly.plot.bar(stacked=True, ax=ax[0, 0], color=["#e50914", "#221f1f"]); ax[0, 0].set_title("Titles added per year (2021 = Jan-Sep only)"); ax[0, 0].set_xlabel("")
bt = by_theme.sort_values("mean").tail(8)
ax[0, 1].barh(bt.index, bt["mean"], color="#e50914"); ax[0, 1].axvline(base, ls="--", c="grey"); ax[0, 1].set_title("Renewal rate by theme (dashed = overall)")
bc = by_country.sort_values("mean").tail(8)
ax[0, 2].barh(bc.index, bc["mean"], color="#221f1f"); ax[0, 2].axvline(base, ls="--", c="grey"); ax[0, 2].set_title("Renewal rate by country (≥40 shows)")
imp_pretty.head(10).sort_values().plot.barh(ax=ax[1, 0], color="#e50914"); ax[1, 0].set_title("Model drivers (permutation importance, AUC drop)")
ax[1, 1].plot(by_year.index, by_year.values, "o-", c="#e50914"); ax[1, 1].set_title("Renewal rate by year added"); ax[1, 1].set_ylim(0, .6)
ax[1, 2].hist(tv.loc[tv["success"] == 1, "p_success"], bins=25, alpha=.7, color="#e50914", label="Renewed")
ax[1, 2].hist(tv.loc[tv["success"] == 0, "p_success"], bins=25, alpha=.6, color="#221f1f", label="1 season")
ax[1, 2].set_title("Predicted success score (out-of-fold)"); ax[1, 2].legend()
save_fig("task6_dashboard.png")

# %%
def md_bold(s):  # **x** -> <b>x</b>
    parts = s.split("**"); return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
img = base64.b64encode(open(f"{OUT}/task6_dashboard.png", "rb").read()).decode()
html = f"""<!doctype html><meta charset="utf-8"><title>Netflix Content Success Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222}}
h1{{color:#e50914}}li{{margin:8px 0}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}</style>
<h1>Netflix Content Success Analytics Engine</h1>
<p>Success proxy: TV show runs 2+ seasons. Generated automatically from <code>netflix_titles.csv</code>.</p>
<h2>Key insights</h2><ul>{''.join(f'<li>{md_bold(i)}</li>' for i in insights)}</ul>
<img src="data:image/png;base64,{img}" alt="dashboard">
<h2>Renewal-profile matches: 1-season shows that look most like renewed shows</h2><p>A profile match from out-of-fold scores, not a prediction of Netflix\'s real decision; some of these may be limited series or already cancelled.</p>{cands.to_html(index=False)}
<h2>Model comparison</h2>{res.to_html()}
<p><i>Caveat: renewal is a proxy; no viewership data exists. Recent shows may simply not have been renewed yet.</i></p>"""
open(f"{OUT}/task6_report.html", "w", encoding="utf-8").write(html)
open(f"{OUT}/task6_insights.md", "w", encoding="utf-8").write("# Automated insights\n\n" + "\n".join("- " + i for i in insights))
print("[saved] outputs/task6_report.html and outputs/task6_insights.md")

# %% [markdown]
# ## MLOps fundamentals – persist the model and reuse it
# The fitted pipeline (preprocessing + model) is saved with `joblib`, with a metadata file for reproducibility.
# `predict_success.py` loads it and scores a new show; no retraining needed.

# %%
final.fit(X, y)  # refit on all data for deployment (metrics above came from held-out data)
os.makedirs("models", exist_ok=True)
joblib.dump({"pipeline": final, "features": FEATS, "themes": themes_kept, "top_countries": list(top_c),
             "director_counts": dir_counts.to_dict()}, "models/success_model.joblib")
json.dump({"model": best_family, "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in gs.best_params_.items()},
           "test_roc_auc": round(float(roc_auc_score(yte, prob)), 4), "n_train_rows": int(len(X)),
           "target": "TV show has 2+ seasons"}, open("models/model_card.json", "w"), indent=2)
print("[saved] models/success_model.joblib, models/model_card.json")

# %% [markdown]
# ## Limitations (read before quoting any number)
# * **Proxy target.** 2+ seasons ≠ audience success; cancellations, limited series and licensing decisions all blur it.
# * **Censoring.** Recently released shows have had little time to be renewed, so they look like failures.
# * **Metadata only.** Without cast, budget, marketing or viewing hours, accuracy has a ceiling (AUC well below 1).
# * **Association, not causation.** "Genre X is renewed more" does not mean adding genre X causes renewal.
