"""Score a new TV show with the saved Task 6 model (no retraining).

Example:
    python predict_success.py
    python predict_success.py '{"title":"My Show","release_year":2021,"year_added":2021,"country":"United States",
        "rating":"TV-MA","themes":["Drama","Crime"],"director":"Jane Doe","month_added":6}'
"""
import sys, json
import joblib
import pandas as pd

bundle = joblib.load("models/success_model.joblib")
pipe, feats, themes, top_c, dcounts = (bundle[k] for k in ("pipeline", "features", "themes", "top_countries", "director_counts"))

RATING_GROUP = {"TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids", "TV-PG": "Teens", "PG": "Teens",
                "TV-14": "Teens", "PG-13": "Teens", "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults"}


def to_row(t: dict) -> pd.DataFrame:
    countries = [c.strip() for c in str(t.get("country", "")).split(",") if c.strip()]
    primary = countries[0] if countries else "Unknown"
    director = t.get("director")
    dfreq = 0 if not director else max(dcounts.get(x.strip(), 0) for x in director.split(","))
    lag = max(-1, min(15, t["year_added"] - t["release_year"]))
    row = {"release_year": t["release_year"], "year_added": t["year_added"], "lag_years": lag,
           "month_added": t.get("month_added", 6), "n_themes": len(t["themes"]), "n_countries": len(countries),
           "director_freq": dfreq, "title_words": len(t["title"].split()),
           "is_coproduction": int(len(countries) > 1), "director_known": int(bool(director)), "fast_add": int(lag <= 1),
           "has_colon": int(":" in t["title"]), "has_number": int(any(ch.isdigit() for ch in t["title"])),
           "country_grp": primary if primary in top_c else "Other", "audience": RATING_GROUP.get(t.get("rating"), "Adults")}
    for th in themes:
        row["th_" + th] = int(th in t["themes"])
    return pd.DataFrame([row])[feats]


if __name__ == "__main__":
    show = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "title": "Example: Night Shift", "release_year": 2021, "year_added": 2021, "country": "United States",
        "rating": "TV-MA", "themes": ["Drama", "Crime", "Thriller"], "director": None, "month_added": 6}
    p = pipe.predict_proba(to_row(show))[0, 1]
    print(f"'{show['title']}': predicted probability of running 2+ seasons = {p:.1%}")
