# Viva preparation — Air Quality Prediction

1. **What is the target?** `CO(GT)`, the reference analyzer's carbon monoxide concentration, in mg/m³.
2. **Is this classification or regression?** Regression, because the target is continuous.
3. **Is this forecasting future CO?** No. This project estimates CO for the same hour using sensor/weather readings available for that hour. Future forecasting would require a lagged/horizon-based design.
4. **Why remove -200?** The UCI data documentation uses -200 as a missing-value code; it is not a real concentration.
5. **Why exclude NMHC(GT)?** It has extremely high missingness, making it unreliable as a primary predictor.
6. **Why exclude C6H6(GT), NOx(GT), NO2(GT)?** The novelty-oriented primary experiment focuses on sensor responses and weather rather than relying on other reference-analyzer pollutant measurements.
7. **Why chronological split?** It simulates evaluation on later observations and avoids randomly mixing earlier and later records.
8. **Why train/validation/test?** Train estimates model parameters, validation selects hyperparameters, and test provides one final held-out evaluation.
9. **Why impute missing predictors?** Many algorithms cannot handle NA directly. Median imputation is robust and learned only from training data.
10. **Why scale KNN/SVR?** Distance- and margin-based methods are sensitive to feature scale. Scaling prevents large-magnitude features from dominating.
11. **What does MAE mean?** Average absolute prediction error in the target's units (mg/m³); lower is better.
12. **What does RMSE mean?** Square root of average squared error; it penalizes large misses more strongly.
13. **What does R² mean?** Relative explained variation compared with predicting the test-set mean; it can be negative if predictions are worse than that baseline.
14. **Why include linear regression?** An interpretable baseline.
15. **KNN vs Random Forest?** KNN predicts from nearby examples; Random Forest averages many randomized trees.
16. **SVR?** It fits a regression function using a margin/epsilon-insensitive loss; this project uses a linear kernel to keep computation manageable.
17. **Gradient Boosting?** Builds trees sequentially, each stage correcting residual patterns from earlier stages.
18. **What is the novelty?** A sensor-and-weather-only predictor set, chronological holdout, comparison of five model families, and a separate high-CO error audit.
19. **What are the limitations?** Single location, historical data (2004–2005), sensor drift, missingness, same-hour estimation rather than forecasting, and limited external validity.
20. **Can the model diagnose air quality or health risk?** No. It is an academic estimate of CO concentration, not a regulatory monitor or medical tool.
21. **What should you show in RStudio?** Import, cleaning, split logic, model fitting, `outputs/test_metrics.csv`, predicted-vs-actual plot, and high-pollution audit.
22. **What if R scores differ from the reference table?** That is expected: the report's reference table is from analogous Python implementations. The final submitted results must come from the included R script.
