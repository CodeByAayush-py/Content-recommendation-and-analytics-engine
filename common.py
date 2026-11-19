"""Shared data loading / cleaning used by every task."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:  # plain `python script.py` -> headless; in Jupyter keep inline plots
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA = os.path.join(ROOT, "data", "netflix_titles.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# Maturity ratings grouped into audience categories (used in Task 3)
RATING_GROUP = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Teens", "PG": "Teens", "TV-14": "Teens", "PG-13": "Teens",
    "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults",
}


# Netflix genre labels are type-specific ("Dramas" only on movies, "TV Dramas" only on shows), which leaks the
# Movie/TV label. THEME_MAP folds them into type-neutral themes for Tasks 2-3.
THEME_MAP = {
    "Action & Adventure": "Action & Adventure", "TV Action & Adventure": "Action & Adventure",
    "Anime Features": "Anime", "Anime Series": "Anime",
    "British TV Shows": "British", "Korean TV Shows": "Korean", "Spanish-Language TV Shows": "Spanish-Language",
    "Children & Family Movies": "Kids & Family", "Kids' TV": "Kids & Family", "Teen TV Shows": "Teen",
    "Classic & Cult TV": "Classic & Cult", "Classic Movies": "Classic & Cult", "Cult Movies": "Classic & Cult",
    "Comedies": "Comedy", "TV Comedies": "Comedy",
    "Crime TV Shows": "Crime",
    "Documentaries": "Documentary", "Docuseries": "Documentary", "Science & Nature TV": "Documentary",
    "Dramas": "Drama", "TV Dramas": "Drama",
    "Faith & Spirituality": "Faith & Spirituality",
    "Horror Movies": "Horror", "TV Horror": "Horror",
    "Independent Movies": "Independent",
    "International Movies": "International", "International TV Shows": "International",
    "LGBTQ Movies": "LGBTQ", "Movies": "Other", "TV Shows": "Other",
    "Music & Musicals": "Music", "Reality TV": "Reality",
    "Romantic Movies": "Romance", "Romantic TV Shows": "Romance",
    "Sci-Fi & Fantasy": "Sci-Fi & Fantasy", "TV Sci-Fi & Fantasy": "Sci-Fi & Fantasy",
    "Sports Movies": "Sports", "TV Mysteries": "Mystery",
    "Stand-Up Comedy": "Stand-Up & Talk", "Stand-Up Comedy & Talk Shows": "Stand-Up & Talk",
    "Thrillers": "Thriller", "TV Thrillers": "Thriller",
}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # 1. duplicates (same title, type and year appear twice for a few rows)
    df = df.drop_duplicates(subset=["title", "type", "release_year"]).reset_index(drop=True)
    # 2. placeholders -> NaN-like flags
    df["director"] = df["director"].replace("Not Given", np.nan)
    df["country"] = df["country"].replace("Not Given", np.nan)
    df["has_director"] = df["director"].notna().astype(int)
    df["primary_country"] = df["country"].fillna("Unknown").str.split(",").str[0].str.strip()
    # 3. dates
    df["date_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y", errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    # 4. duration -> number + unit
    df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = df["duration"].str.extract(r"\d+\s+(\w+)")
    # 5. genres
    df["genres"] = df["listed_in"].str.split(",").apply(lambda xs: [x.strip() for x in xs])
    df["themes"] = df["genres"].apply(lambda gs: sorted({THEME_MAP[g] for g in gs}))
    df["n_genres"] = df["genres"].str.len()
    # 6. audience group
    df["audience"] = df["rating"].map(RATING_GROUP)
    return df


def genre_matrix(df: pd.DataFrame, min_count: int = 1, col: str = "genres") -> pd.DataFrame:
    """Multi-hot genre columns (prefixed 'g_'). col='themes' gives the type-neutral version."""
    exploded = df[col].explode()
    keep = exploded.value_counts()
    keep = keep[keep >= min_count].index
    m = pd.get_dummies(exploded[exploded.isin(keep)]).groupby(level=0).max().astype(int)
    m.columns = ["g_" + c for c in m.columns]
    return m.reindex(df.index, fill_value=0)


def save_fig(name: str):
    path = os.path.join(OUT, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[saved] {path}")
