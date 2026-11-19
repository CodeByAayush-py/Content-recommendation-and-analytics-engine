# %% [markdown]
# # Task 4 (Medium) – Netflix Content Segmentation (Unsupervised)
# Cluster titles with K-Means (primary) and Agglomerative/Ward (cross-check), scale features, visualise with PCA,
# and interpret what each cluster represents.

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import load_clean, genre_matrix, save_fig, RATING_GROUP

df = load_clean()
df = df[df["audience"].notna()].reset_index(drop=True)  # drop NR / UR

# %% [markdown]
# ## Step 1 – Prepare numerical and categorical features
# * Numerical (standardised): release year, year added, duration *z-scored within type* (minutes vs seasons are not
#   comparable), number of genres.
# * Categorical (0/1): type, genre themes (≥50 titles), audience group, top-8 countries.
# Binary columns are kept 0/1 (not standardised) so rare categories do not dominate the distance.

# %%
G = genre_matrix(df, min_count=50, col="themes").drop(columns=["g_Other"], errors="ignore")
top_c = df["primary_country"].value_counts().head(8).index
C = pd.get_dummies(df["primary_country"].where(df["primary_country"].isin(top_c), "Other"), prefix="c")
aud = pd.get_dummies(df["audience"], prefix="a")
dur = df.groupby("type")["duration_value"].transform(lambda s: (s - s.mean()) / s.std())

num = pd.DataFrame({"release_year": df["release_year"], "year_added": df["year_added"].fillna(df["year_added"].median()),
                    "duration_z": dur, "n_genres": df["n_genres"]})
num_s = pd.DataFrame(StandardScaler().fit_transform(num), columns=num.columns)
cat = pd.concat([(df["type"] == "TV Show").astype(int).rename("is_tv"), G, aud, C], axis=1).astype(float)
X = pd.concat([num_s, cat], axis=1)
# Movies and TV shows are structurally different formats, so `is_tv` gets 3x weight. Without it the clusters mix both
# formats and the silhouette is lower (k=6: 0.11 -> ~0.15 with weight 3; tested weights 1, 2, 3).
X["is_tv"] *= 3
print("Feature matrix:", X.shape)

# %% [markdown]
# ## Step 2 – Apply clustering algorithms
# Choose k with the elbow (inertia) and silhouette score for k = 2…10.

# %%
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_, sample_size=4000, random_state=42))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ks, inertia, "o-", c="#e50914"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", c="#221f1f"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
save_fig("task4_choose_k.png")
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia), "silhouette": np.round(sil, 3)}).to_string(index=False))

# %%
# k=2 scores highest but is only the Movie/TV split (already known). We want finer content segments, so we take the
# best-scoring k >= 3 whose clusters stay interpretable: k=5.
K = 5
km = KMeans(K, n_init=20, random_state=42).fit(X)
df["cluster"] = km.labels_
agg = AgglomerativeClustering(K, linkage="ward").fit(X)
print(f"K-Means silhouette = {silhouette_score(X, km.labels_, sample_size=4000, random_state=42):.3f}")
print(f"Ward silhouette    = {silhouette_score(X, agg.labels_, sample_size=4000, random_state=42):.3f}")
print(f"Agreement between the two algorithms (Adjusted Rand Index) = {adjusted_rand_score(km.labels_, agg.labels_):.3f}")

# %% [markdown]
# ## Step 3 – Identify content groups (cluster profiles)

# %%
def top(series_of_lists, n=3):
    return ", ".join(series_of_lists.explode().value_counts().head(n).index)

rows = []
for c, g in df.groupby("cluster"):
    rows.append({"cluster": c, "size": len(g), "% TV": round(100 * (g["type"] == "TV Show").mean(), 1),
                 "median release yr": int(g["release_year"].median()),
                 "median duration": f'{g["duration_value"].median():.0f} {"min" if (g["type"]=="Movie").mean()>.5 else "seasons"}',
                 "top audience": g["audience"].value_counts().index[0],
                 "top themes": top(g["themes"], 3), "top countries": ", ".join(g["primary_country"].value_counts().head(2).index)})
prof = pd.DataFrame(rows)
print(prof.to_string(index=False))

# %% [markdown]
# ## Step 4 – Visualise clusters (PCA to 2D)

# %%
pca = PCA(2, random_state=42); Z = pca.fit_transform(X)
colors = ["#e50914", "#221f1f", "#0071eb", "#f5a623", "#2ca02c", "#9467bd", "#8c564b"]
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(K):
    m = df["cluster"] == c
    ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.5, c=colors[c], label=f"Cluster {c} (n={m.sum()})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("Netflix content clusters – PCA projection"); ax.legend(markerscale=3, fontsize=8)
save_fig("task4_clusters_pca.png")

# %% [markdown]
# ## Step 5 – Interpret cluster characteristics
# Name each cluster from its dominant traits (we print the evidence, then label by hand).

# %%
ex = df[["cluster", "themes"]].explode("themes").reset_index(drop=True)
share = pd.crosstab(ex["cluster"], ex["themes"], normalize="index")
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(share.values, cmap="Reds", aspect="auto")
ax.set_yticks(range(K)); ax.set_yticklabels([f"Cluster {c}" for c in share.index])
ax.set_xticks(range(share.shape[1])); ax.set_xticklabels(share.columns, rotation=60, ha="right", fontsize=8)
ax.set_title("Genre-theme mix per cluster (share of theme tags)"); plt.colorbar(im)
save_fig("task4_cluster_themes.png")

print("\nExample titles per cluster:")
for c in range(K):
    print(f"Cluster {c}:", "; ".join(df[df.cluster == c].sample(5, random_state=1)["title"]))
# Human-readable names, derived from the profile table above (cluster ids are stable because random_state is fixed)
NAMES = {0: "Adult international TV dramas & crime", 1: "Everyday movies: documentaries, family & stand-up",
         2: "Mainstream international movies (drama/comedy, teen-adult)", 3: "Classic older movies (pre-2000s)",
         4: "Kids & family TV series"}
prof.insert(1, "name", prof["cluster"].map(NAMES))
df["cluster_name"] = df["cluster"].map(NAMES)
print(prof[["cluster", "name", "size"]].to_string(index=False))
prof.to_csv("outputs/task4_cluster_profiles.csv", index=False)
df[["show_id", "title", "type", "cluster", "cluster_name"]].to_csv("outputs/task4_cluster_assignments.csv", index=False)
