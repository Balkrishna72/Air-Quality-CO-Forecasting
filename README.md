# Air Quality Prediction Using Machine Learning in R

## Project at a glance
- **Task:** Same-hour estimation of carbon monoxide (CO) concentration, `CO(GT)` in mg/m³.
- **Dataset:** UCI Air Quality dataset (2004–2005, Italian city; 9,358 documented hourly observations).
- **Novelty angle:** Sensor-and-weather-only predictors, chronological validation, five-model comparison, and a high-pollution error audit.
- **Models:** Multiple Linear Regression, KNN Regression, Support Vector Regression (linear kernel), Random Forest, Gradient Boosting.
- **Evaluation:** MAE, RMSE, R² on a chronologically held-out test period.

## Important scope note
This is **same-hour estimation**, not a claim that the model forecasts tomorrow's or next-hour CO. A true future forecast would need lagged predictors and a different target horizon. The primary experiment intentionally excludes other reference-analyzer pollutant columns (`C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) and excludes `NMHC(GT)` because of its extreme missingness. It uses five sensor response variables, weather variables, and calendar features.

## Run in RStudio
1. Extract the ZIP.
2. Open `Air_Quality_Prediction.Rproj`.
3. Ensure `data/AirQualityUCI.csv` is present.
4. Open `scripts/run_project.R`.
5. Run `source("scripts/run_project.R")` from the project root.
6. The script installs missing packages and creates `outputs/test_metrics.csv`, `outputs/test_predictions.csv`, `outputs/high_pollution_error_audit.csv`, and figures under `outputs/figures/`.

## Project structure
- `data/AirQualityUCI.csv` — original user-provided data.
- `data/cleaned_target_valid.csv` — reproducible cleaned modeling table snapshot.
- `scripts/run_project.R` — end-to-end R workflow.
- `outputs/reference_model_metrics.csv` — **reference benchmark only**, generated with analogous Python model families because R was not available in the build environment. Run the R script to produce the authoritative R results.
- `outputs/reference_test_predictions.csv` — corresponding reference predictions, not the final R run.
- `outputs/figures/` — exploratory and reference benchmark figures.
- `report/Project_Report.docx` and `.pdf` — project report.
- `docs/Viva_Questions.md` — viva preparation.
- `docs/LinkedIn_Post_Draft.md` — editable post draft.
- `docs/Rubric_Checklist.md` — submission checklist.

## Data handling
- The CSV uses `;` separators and decimal commas.
- Entirely blank rows and trailing empty columns are removed.
- `-200` is treated as a missing measurement code, not a real value.
- Rows without a valid target are excluded from supervised model training.
- Predictor imputation and scaling are learned from the training partition to avoid test leakage.
- Chronological split: first 70% train, next 15% validation, final 15% test.

## Reference benchmark (not the final R output)
| Model                                     |   MAE_mg_m3 |   RMSE_mg_m3 |        R2 |
|:------------------------------------------|------------:|-------------:|----------:|
| Random Forest Regression                  |    0.322516 |     0.48483  |  0.855264 |
| Gradient Boosting Regression              |    0.324199 |     0.485762 |  0.854707 |
| Support Vector Regression (linear kernel) |    0.335622 |     0.494823 |  0.849237 |
| KNN Regression                            |    0.39213  |     0.554763 |  0.810499 |
| Multiple Linear Regression                |    1.31533  |     1.40575  | -0.216777 |

These values are from an analogous Python benchmark and are included as a sanity-check reference only. The R implementations and package defaults can produce different values. Re-run the R script and use `outputs/test_metrics.csv` as the final result in any submission.

## Dataset citation
Vito, S. (2008). *Air Quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
