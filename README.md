# Netflix Machine Learning Internship – Auspify Technologies

All six tasks completed on `data/netflix_titles.csv` (8,790 titles, 10 columns; 8,787 after removing 3 duplicates).

| # | Task | Difficulty | Headline result |
|---|------|-----------|-----------------|
| 1 | Content Recommendation System | Easy | TF-IDF + cosine similarity; recovers a hidden genre in the top-5 **29%** of the time vs **14%** by chance |
| 2 | Content Type Prediction (Movie vs TV) | Easy | **97.2%** accuracy / 0.953 F1 (Gradient Boosting), majority baseline 69.7% |
| 3 | Audience Rating Classification | Medium | **73.2%** accuracy / 0.756 macro-F1 (tuned Gradient Boosting), majority baseline 46.0% |
| 4 | Content Segmentation | Medium | K-Means, k=5 → 5 interpretable segments (silhouette 0.174; Ward agrees, ARI 0.60) |
| 5 | Trend Forecasting | Advanced | Random Forest on lag features: MAE 35 titles/month (≈21% MAPE) on a 12-month hold-out vs 45 for seasonal naive; forecast ≈1,930 titles for Oct 2021–Sep 2022 |
| 6 | Content Success Analytics Engine | Advanced | End-to-end pipeline; predicts TV renewal (2+ seasons) with ROC-AUC **0.76** (tuned Random Forest) vs 0.50 baseline; auto-generated insights, HTML report, saved model |

## Project layout
```
common.py                      shared loading / cleaning (run by every task)
task1_recommender.py/.ipynb
task2_type_prediction.py/.ipynb
task3_rating_classification.py/.ipynb
task4_segmentation.py/.ipynb
task5_trend_forecasting.py/.ipynb
task6_success_engine.py/.ipynb
predict_success.py             scores a new show with the saved Task 6 model
models/                        saved model (joblib) + model card
make_notebooks.py              regenerates the .ipynb files from the .py files
data/netflix_titles.csv
outputs/                       charts (PNG), metrics (CSV/TXT), run logs
```

## How to run
```bash
pip install -r requirements.txt
python task1_recommender.py          # or open task1_recommender.ipynb in Jupyter / Colab
python task2_type_prediction.py
python task3_rating_classification.py   # ~3-5 min (grid search)
python task4_segmentation.py
python task5_trend_forecasting.py
python task6_success_engine.py          # ~1-2 min; creates models/ and outputs/task6_report.html
python predict_success.py               # demo: score a new show with the saved model
```
Run from the project folder. Charts are written to `outputs/`. In Jupyter they also display inline – take your
"screenshots of completed tasks" from the notebooks.

## Key findings and honest caveats
* **Dataset has no plot descriptions or viewer scores.** The recommender therefore uses metadata only
  (genres, director, country, rating, type), and "audience rating" in Task 3 means the *maturity rating* grouped into
  Kids / Teens / Adults (NR/UR excluded).
* **Task 2 – data leakage.** `duration` ("90 min" / "2 Seasons") predicts the type 100%, and *every one* of the 42
  genre labels exists for only one type ("Dramas" = movie, "TV Dramas" = show). Using them gives 99.6–100% accuracy,
  which is meaningless. The reported model uses type-neutral genre themes and no duration. The remaining strongest
  signal is whether a director is listed (97% of movies vs 9% of shows); without it accuracy is 89.4%.
  (`outputs/task2_leakage_demo.csv`)
* **Task 3** uses macro-F1 for tuning/selection (Kids is a 10% minority) and picks the final model by
  cross-validation, not by the test set. Predicting the exact 12-class rating is much harder (57% vs 37% baseline).
* **Task 4** weights Movie/TV 3x because they are different formats; k=2 has the best silhouette but only re-discovers
  the Movie/TV split, so k=5 was chosen. Silhouette values are modest (typical for mostly-binary features).

* **Task 5** forecasts monthly titles *added* (`date_added`), starting Jan 2016 because earlier months have only a few
  titles. Models are compared on a time-ordered 12-month hold-out plus a 4-origin rolling backtest and the final model
  is picked on the backtest. Growth plateaued after 2019, so the log-linear trend model (MAPE >100%) wildly overshoots –
  a useful lesson in why extrapolating early growth fails. Forecasts are flat-ish with wide uncertainty (80% band
  ≈ ±25%); the dataset ends Sep 2021 and is a sample of the catalogue, so treat the numbers as illustrative. The
  per-type bonus chart fits separate models, so its totals do not sum exactly to the overall forecast.

* **Task 6** needs a definition of "success", but the data has no viewership. The proxy is **TV show renewal
  (2+ seasons)**, so movies appear only in the descriptive analysis. Features exclude `duration` (it *is* the target).
  Preprocessing sits inside sklearn Pipelines, so cross-validation is leak-free. Test ROC-AUC 0.76 (baseline 0.50) means
  moderate signal, not certainty. Biggest drivers: country, the International (22% renewal) and Documentary (16%) themes, which
  renew below the 33% average, then year added and Kids & Family (51%). Limits: proxy target, recent shows may not be renewed yet,
  metadata only, association not causation. The "renewal-profile matches" list is a profile match, not a prediction of
  Netflix's real decisions. Open `outputs/task6_report.html` in a browser for the full visual report.

## Segments found (Task 4)
| Cluster | Name | Size |
|---|---|---|
| 0 | Adult international TV dramas & crime | 1,568 |
| 1 | Everyday movies: documentaries, family & stand-up | 2,371 |
| 2 | Mainstream international movies (drama/comedy) | 3,143 |
| 3 | Classic older movies (pre-2000s) | 553 |
| 4 | Kids & family TV series | 1,070 |

## Submission checklist (Auspify)
- [ ] Push this folder to a public GitHub repo and share the link
- [ ] Run the notebooks and screenshot the key outputs for each task
- [ ] Optional: 2–3 minute demo video walking through the notebooks
- [ ] Post on LinkedIn with #Auspify #AuspifyTechnologies #AuspifyInternship #AuspifyProjects
