# Automated insights

- **Catalogue growth:** titles added per year peaked in 2019 (2,014 titles). TV's share of additions moved from 41% (2016) to 25% (2018) to 34% (2021).
- **Where content comes from:** United States (37% of titles), then India (12%) and United Kingdom (7%).
- **Base rate:** 33% of TV shows in the catalogue ran for 2+ seasons.
- **Genres that get renewed:** Teen (57%, n=69) and Sci-Fi & Fantasy (54%) lead; Romance (16%) and Documentary (16%) lag.
- **Countries (≥40 shows):** best renewal rate is Canada (67%), lowest is Taiwan (1%).
- **Audience:** Kids 48% (n=464), Adults 32% (n=1145), Teens 26% (n=1050) renewal rate.
- **Model:** tuned Random Forest separates renewed from one-season shows with test ROC-AUC 0.76 (0.50 = chance; out-of-fold on all shows 0.76). Top drivers: country_grp, theme: International, theme: Documentary, year_added.
- **Timing:** the renewal rate of shows added in 2021 is 42% vs 21% for 2016 additions. The data cannot say why (catalogue mix, licensing and sampling all change by year), so the model only uses it as a weak signal.