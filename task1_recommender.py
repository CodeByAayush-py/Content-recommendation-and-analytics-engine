# %% [markdown]
# # Task 1 (Easy) – Netflix Content Recommendation System
# Content-based recommender: TF-IDF over genres, director, country, rating and type + cosine similarity.
# The dataset has no plot descriptions, so recommendations rely on the metadata columns.

# %%
import numpy as np, pandas as pd, re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from common import load_clean, save_fig

df = load_clean()
print(df.shape)

# %% [markdown]
# ## Step 1 – Prepare content features
# Each title becomes a "document" of tokens. Multi-word values (e.g. *International Movies*) are glued with `_`
# so they stay one token. Genres are repeated 2x so they weigh more than director/country.

# %%
def tok(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def soup(row):
    genres = [("g_" + tok(g)) for g in row["genres"]]
    directors = [("d_" + tok(d)) for d in str(row["director"]).split(",")] if pd.notna(row["director"]) else []
    countries = [("c_" + tok(c)) for c in str(row["country"]).split(",")] if pd.notna(row["country"]) else []
    return " ".join(genres * 2 + directors + countries + ["r_" + tok(row["rating"]), "t_" + tok(row["type"])])

df["soup"] = df.apply(soup, axis=1)
print(df["soup"].iloc[0])

# %% [markdown]
# ## Step 2 – Convert text to machine-readable format (TF-IDF)

# %%
tfidf = TfidfVectorizer(token_pattern=r"\S+")
X = tfidf.fit_transform(df["soup"])
print("TF-IDF matrix:", X.shape)

# %% [markdown]
# ## Step 3 – Content similarity scores (cosine). `linear_kernel` = cosine on L2-normalised TF-IDF rows.

# %%
def similarities(i):
    return linear_kernel(X[i], X).ravel()

title_index = pd.Series(df.index, index=df["title"].str.lower())

# %% [markdown]
# ## Step 4 – Generate recommendations

# %%
def recommend(title, k=5):
    key = title.lower()
    if key not in title_index.index:
        raise KeyError(f"'{title}' not found")
    i = title_index[key]
    if isinstance(i, pd.Series):  # duplicate titles
        i = i.iloc[0]
    sims = similarities(i)
    sims[i] = -1
    top = np.argsort(-sims)[:k]
    out = df.loc[top, ["title", "type", "release_year", "rating", "listed_in"]].copy()
    out.insert(0, "similarity", sims[top].round(3))
    return df.loc[i, ["title", "type", "listed_in"]].tolist(), out

samples = ["Ganglands", "Midnight Mass", "The Great British Baking Show", "Motu Patlu in Wonderland",
           "Sankofa", "Dick Johnson Is Dead"]
lines = []
for s in samples:
    src, rec = recommend(s)
    print(f"\n=== Because you watched: {src[0]} ({src[1]}; {src[2]}) ===")
    print(rec.to_string(index=False))
    lines.append(f"=== {s} ===\n{rec.to_string(index=False)}\n")
open("outputs/task1_sample_recommendations.txt", "w").write("\n".join(lines))

# %% [markdown]
# ## Step 5 – Evaluate recommendation quality
# There are no user ratings, so we use two checks:
# 1. **Genre-overlap@5 vs a random baseline** (mean Jaccard of genres between a title and its 5 recommendations).
# 2. **Hidden-genre recovery** (a real hold-out test): for each sampled title we *hide one of its genres*,
#    rebuild its vector without it, and measure what share of the top-5 recommendations still carry the hidden genre,
#    compared with how common that genre is overall (random chance).

# %%
rng = np.random.default_rng(42)
genre_sets = df["genres"].apply(set)

def jaccard(a, b):
    return len(a & b) / len(a | b)

sample_ids = rng.choice(len(df), 600, replace=False)
rec_j, rand_j = [], []
for i in sample_ids:
    sims = similarities(i); sims[i] = -1
    top = np.argsort(-sims)[:5]
    rec_j.append(np.mean([jaccard(genre_sets[i], genre_sets[j]) for j in top]))
    rnd = rng.choice(len(df), 5, replace=False)
    rand_j.append(np.mean([jaccard(genre_sets[i], genre_sets[j]) for j in rnd]))
print(f"Genre Jaccard@5  recommender: {np.mean(rec_j):.3f} | random: {np.mean(rand_j):.3f}")

# hidden-genre recovery
multi = [i for i in range(len(df)) if len(df.at[i, "genres"]) >= 2]
test_ids = rng.choice(multi, 500, replace=False)
hits, chance = [], []
for i in test_ids:
    gs = df.at[i, "genres"]
    hide = gs[rng.integers(len(gs))]
    row = df.loc[i].copy()
    row["genres"] = [g for g in gs if g != hide]
    vec = tfidf.transform([soup(row)])
    sims = linear_kernel(vec, X).ravel(); sims[i] = -1
    top = np.argsort(-sims)[:5]
    hits.append(np.mean([hide in genre_sets[j] for j in top]))
    chance.append(np.mean([hide in s for s in genre_sets]))
print(f"Hidden-genre recovery@5  recommender: {np.mean(hits):.3f} | chance: {np.mean(chance):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(["Random", "Recommender"], [np.mean(rand_j), np.mean(rec_j)], color=["#9aa0a6", "#e50914"])
ax[0].set_title("Genre overlap (Jaccard) @5\n(optimistic: genres are model inputs)", fontsize=10); ax[0].set_ylim(0, 1)
ax[1].bar(["Chance", "Recommender"], [np.mean(chance), np.mean(hits)], color=["#9aa0a6", "#e50914"])
ax[1].set_title("Hidden-genre recovery @5\n(hold-out test)", fontsize=10); ax[1].set_ylim(0, 1)
for a in ax:
    for p in a.patches:
        a.text(p.get_x() + p.get_width() / 2, p.get_height() + .02, f"{p.get_height():.2f}", ha="center")
save_fig("task1_evaluation.png")

with open("outputs/task1_metrics.txt", "w") as f:
    f.write(f"Genre Jaccard@5 recommender={np.mean(rec_j):.3f} random={np.mean(rand_j):.3f}\n")
    f.write(f"Hidden-genre recovery@5 recommender={np.mean(hits):.3f} chance={np.mean(chance):.3f}\n")
