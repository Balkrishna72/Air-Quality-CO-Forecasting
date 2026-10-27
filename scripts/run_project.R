# ============================================================
# Air Quality Prediction: Time-aware comparative regression
# Course: Predictive Analytics | Language: R
# ============================================================
# Run from the project root in RStudio:
# source("scripts/run_project.R")
# The script reads data/AirQualityUCI.csv and writes outputs/.
# It is designed to be reproducible; results may differ slightly
# from the Python reference benchmark included in the report.
# ============================================================

required <- c("readr", "dplyr", "tidyr", "ggplot2", "e1071", "randomForest", "gbm", "FNN")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
invisible(lapply(required, library, character.only = TRUE))

dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# 1. Import: source file uses semicolon delimiters and comma decimals.
raw <- readr::read_delim(
  "data/AirQualityUCI.csv", delim = ";",
  locale = readr::locale(decimal_mark = ","),
  na = c("", "NA"), trim_ws = TRUE,
  show_col_types = FALSE, name_repair = "unique"
)

# Remove trailing empty/unnamed columns and fully blank rows.
keep_cols <- !grepl("^\\.\\.\\.[0-9]+$", names(raw)) &
  !grepl("^Unnamed", names(raw))
raw <- raw[, keep_cols, drop = FALSE]
raw <- raw[rowSums(!is.na(raw)) > 0, , drop = FALSE]

# Convert measurements to numeric safely, then translate -200 to NA.
measure_cols <- setdiff(names(raw), c("Date", "Time"))
for (nm in measure_cols) {
  raw[[nm]] <- suppressWarnings(as.numeric(gsub(",", ".", as.character(raw[[nm]]), fixed = TRUE)))
  raw[[nm]][raw[[nm]] == -200] <- NA_real_
}

raw$Timestamp <- as.POSIXct(
  paste(raw$Date, raw$Time),
  format = "%d/%m/%Y %H.%M.%S", tz = "UTC"
)
raw <- raw[!is.na(raw$Timestamp), , drop = FALSE]

raw <- raw[order(raw$Timestamp), , drop = FALSE]
raw$Hour <- as.integer(format(raw$Timestamp, "%H"))
raw$Month <- as.integer(format(raw$Timestamp, "%m"))
raw$DayOfWeek <- as.integer(format(raw$Timestamp, "%u")) - 1L

# Keep only consecutive hourly observations for lag creation
hour_gap <- c(NA_real_,
              as.numeric(diff(raw$Timestamp), units = "hours"))

raw$HourGap <- hour_gap

# Create lagged CO features only across consecutive hourly records
raw$CO_Lag1 <- ifelse(
  raw$HourGap == 1,
  dplyr::lag(raw[["CO(GT)"]], 1),
  NA_real_
)

raw$CO_Lag2 <- ifelse(
  raw$HourGap == 1 &
    dplyr::lag(raw$HourGap, 1) == 1,
  dplyr::lag(raw[["CO(GT)"]], 2),
  NA_real_
)

raw$CO_Lag3 <- ifelse(
  raw$HourGap == 1 &
    dplyr::lag(raw$HourGap, 1) == 1 &
    dplyr::lag(raw$HourGap, 2) == 1,
  dplyr::lag(raw[["CO(GT)"]], 3),
  NA_real_
)


# Forecasting setup: predict CO at t+1 using information available at t.
# Reference analyzer measurements NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)
# are excluded as predictors; current-hour CO and its historical lags are used.

# Current-hour CO + sensor/weather/calendar + historical CO predictors.
# At timestamp t, CO_Current is CO(t), while the target is CO(t+1).
raw$CO_Current <- raw[["CO(GT)"]]
features <- c(
  "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
  "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
  "Hour", "Month", "DayOfWeek", "CO_Current",
  "CO_Lag1", "CO_Lag2", "CO_Lag3"
)

target <- "Target_CO_Next_Hour"
data <- raw[, c("Timestamp", features), drop = FALSE]

# Create the next-hour CO forecasting target

# Create target only when the next observation is exactly one hour later

# Create the next-hour target only for consecutive hourly records
next_timestamp <- dplyr::lead(data$Timestamp, 1)
next_co <- dplyr::lead(raw[["CO(GT)"]], 1)

data$Target_CO_Next_Hour <- ifelse(
  as.numeric(difftime(next_timestamp, data$Timestamp, units = "hours")) == 1,
  next_co,
  NA_real_
)




# Remove rows without a valid next-hour target
data <- data[!is.na(data$Target_CO_Next_Hour), , drop = FALSE]

# Keep rows with an observed current-hour CO value and complete lag history.
# Missing sensor/weather predictors are imputed later using training medians.
data <- data[
  !is.na(data$CO_Current) &
    !is.na(data$CO_Lag1) &
    !is.na(data$CO_Lag2) &
    !is.na(data$CO_Lag3),
  ,
  drop = FALSE
]


# Verify lag-feature and next-hour target alignment
stopifnot(all(!is.na(data$CO_Lag1)))
stopifnot(all(!is.na(data$CO_Lag2)))
stopifnot(all(!is.na(data$CO_Lag3)))
stopifnot(all(!is.na(data$Target_CO_Next_Hour)))

cat("Lag and target checks passed:", nrow(data), "rows\n")

# Check the new forecasting target
head(data[, c("Timestamp", target, "Target_CO_Next_Hour")], 5)

write.csv(data, "data/cleaned_target_valid.csv", row.names = FALSE)
# Verify chronological order and next-hour target alignment
stopifnot(all(diff(data$Timestamp) > 0))
cat("Exactly 1-hour gaps:", 
    sum(diff(data$Timestamp) == 3600), "\n")
cat("Other time gaps:", 
    sum(diff(data$Timestamp) != 3600), "\n")
cat("Valid next-hour forecasting rows:", nrow(data), "\n")
cat("First timestamp:", as.character(min(data$Timestamp)), "\n")
cat("Last timestamp:", as.character(max(data$Timestamp)), "\n")
n <- nrow(data)
i_train <- seq_len(floor(0.70 * n))
i_val <- (max(i_train) + 1L):floor(0.85 * n)
i_test <- (max(i_val) + 1L):n
train <- data[i_train, , drop = FALSE]
valid <- data[i_val, , drop = FALSE]
test <- data[i_test, , drop = FALSE]

# Imputation values are learned from training data only.
get_medians <- function(d, cols) {
  vapply(cols, function(z) {
    m <- median(d[[z]], na.rm = TRUE)
    if (!is.finite(m)) 0 else m
  }, numeric(1))
}
apply_impute <- function(d, cols, meds) {
  out <- d[, cols, drop = FALSE]
  for (z in cols) out[[z]][is.na(out[[z]])] <- meds[[z]]
  out
}
med_train <- get_medians(train, features)
xtr <- apply_impute(train, features, med_train)
xv <- apply_impute(valid, features, med_train)
ytr <- train[[target]]; yv <- valid[[target]]

# Standardization learned from training data only for KNN and SVR.
mu <- vapply(xtr, mean, numeric(1))
sig <- vapply(xtr, sd, numeric(1)); sig[!is.finite(sig) | sig == 0] <- 1
scale_with <- function(x, mu, sig) {
  out <- as.data.frame(scale(x, center = mu, scale = sig))
  names(out) <- names(x)
  out
}
xtr_s <- scale_with(xtr, mu, sig)
xv_s <- scale_with(xv, mu, sig)

rmse <- function(y, p) sqrt(mean((y - p)^2))
mae <- function(y, p) mean(abs(y - p))
r2 <- function(y, p) 1 - sum((y-p)^2) / sum((y-mean(y))^2)
metric_row <- function(name, y, p) data.frame(
  Model = name, MAE_mg_m3 = mae(y,p), RMSE_mg_m3 = rmse(y,p), R2 = r2(y,p)
)

# 2. Fit and tune on chronological training/validation partitions.
# Multiple linear regression baseline
lm_train_data <- data.frame(y = ytr, xtr, check.names = FALSE)
lm_fit <- lm(y ~ ., data = lm_train_data)
lm_val_data <- data.frame(xv, check.names = FALSE)
lm_val <- as.numeric(predict(lm_fit, newdata = lm_val_data))
# KNN: select k by validation RMSE
knn_grid <- c(5, 9, 15, 25)
knn_scores <- sapply(knn_grid, function(k) {
  p <- FNN::knn.reg(train = as.matrix(xtr_s), test = as.matrix(xv_s),
                    y = ytr, k = k)$pred
  rmse(yv, p)
})
best_k <- knn_grid[which.min(knn_scores)]
knn_val <- FNN::knn.reg(train = as.matrix(xtr_s), test = as.matrix(xv_s),
                        y = ytr, k = best_k)$pred

# Linear-kernel Support Vector Regression; cost tuned on validation.
svm_costs <- c(0.1, 1, 10)
svm_scores <- sapply(svm_costs, function(cost_value) {
  fit <- e1071::svm(x = xtr_s, y = ytr, type = "eps-regression",
                    kernel = "linear", cost = cost_value, epsilon = 0.1,
                    scale = FALSE)
  rmse(yv, predict(fit, xv_s))
})
best_cost <- svm_costs[which.min(svm_scores)]
svm_fit <- e1071::svm(x = xtr_s, y = ytr, type = "eps-regression",
                      kernel = "linear", cost = best_cost, epsilon = 0.1,
                      scale = FALSE)
svm_val <- predict(svm_fit, xv_s)

# Random Forest: tune mtry
mtry_grid <- unique(pmax(1, pmin(length(features), c(3, 5, 7))))
rf_scores <- sapply(mtry_grid, function(mtry_value) {
  fit <- randomForest::randomForest(x = xtr, y = ytr, ntree = 300,
                                    mtry = mtry_value, nodesize = 5)
  rmse(yv, predict(fit, xv))
})
best_mtry <- mtry_grid[which.min(rf_scores)]
rf_fit <- randomForest::randomForest(x = xtr, y = ytr, ntree = 300,
                                     mtry = best_mtry, nodesize = 5,
                                     importance = TRUE)
rf_val <- predict(rf_fit, xv)

# Gradient Boosting: select number of trees/depth on validation.
gb_grid <- expand.grid(
  n.trees = c(200, 300, 500, 700, 1000),
  interaction.depth = c(2, 3, 4, 5, 6),
  shrinkage = c(0.01, 0.03, 0.05, 0.1)
)
gb_scores <- numeric(nrow(gb_grid))
for (j in seq_len(nrow(gb_grid))) {
  fit <- gbm::gbm.fit(x = as.matrix(xtr), y = ytr,
                      distribution = "gaussian",
                      n.trees = gb_grid$n.trees[j],
                      interaction.depth = gb_grid$interaction.depth[j],
                      shrinkage = gb_grid$shrinkage[j],
                      bag.fraction = 0.8, verbose = FALSE)
  p <- predict(fit, newdata = as.matrix(xv), n.trees = gb_grid$n.trees[j])
  gb_scores[j] <- rmse(yv, p)
}
gb_choice <- gb_grid[which.min(gb_scores), ]
gb_fit <- gbm::gbm.fit(x = as.matrix(xtr), y = ytr,
                       distribution = "gaussian",
                       n.trees = gb_choice$n.trees,
                       interaction.depth = gb_choice$interaction.depth,
                       shrinkage = gb_choice$shrinkage, n.minobsinnode = 10,
                       bag.fraction = 0.8, verbose = FALSE)
gb_val <- predict(gb_fit, newdata = as.matrix(xv), n.trees = gb_choice$n.trees)

validation_metrics <- dplyr::bind_rows(
  metric_row("Multiple Linear Regression", yv, lm_val),
  metric_row(paste0("KNN Regression (k=", best_k, ")"), yv, knn_val),
  metric_row(paste0("Support Vector Regression (linear, cost=", best_cost, ")"), yv, svm_val),
  metric_row(paste0("Random Forest (mtry=", best_mtry, ")"), yv, rf_val),
  metric_row(paste0("Gradient Boosting (trees=", gb_choice$n.trees,
                    ", depth=", gb_choice$interaction.depth, ")"), yv, gb_val)
)
write.csv(validation_metrics, "outputs/validation_metrics.csv", row.names = FALSE)

# 3. Final fit on train+validation, evaluate once on the untouched test period.
trainval <- data[c(i_train, i_val), , drop = FALSE]
med_tv <- get_medians(trainval, features)
xtv <- apply_impute(trainval, features, med_tv)
xte <- apply_impute(test, features, med_tv)
ytv <- trainval[[target]]; yte <- test[[target]]
mu_tv <- vapply(xtv, mean, numeric(1))
sig_tv <- vapply(xtv, sd, numeric(1)); sig_tv[!is.finite(sig_tv) | sig_tv == 0] <- 1
xtv_s <- scale_with(xtv, mu_tv, sig_tv)
xte_s <- scale_with(xte, mu_tv, sig_tv)


lm_trainval_data <- data.frame(y = ytv, xtv, check.names = FALSE)
lm_final <- lm(y ~ ., data = lm_trainval_data)

lm_test_data <- data.frame(xte, check.names = FALSE)
p_lm <- predict(lm_final, newdata = lm_test_data)

p_knn <- FNN::knn.reg(train = as.matrix(xtv_s), test = as.matrix(xte_s),
                      y = ytv, k = best_k)$pred
svm_final <- e1071::svm(x = xtv_s, y = ytv, type = "eps-regression",
                        kernel = "linear", cost = best_cost, epsilon = 0.1,
                        scale = FALSE)
p_svm <- predict(svm_final, xte_s)
rf_final <- randomForest::randomForest(x = xtv, y = ytv, ntree = 300,
                                       mtry = best_mtry, nodesize = 5,
                                       importance = TRUE)
p_rf <- predict(rf_final, xte)
gb_final <- gbm::gbm.fit(x = as.matrix(xtv), y = ytv,
                         distribution = "gaussian",
                         n.trees = gb_choice$n.trees,
                         interaction.depth = gb_choice$interaction.depth,
                         shrinkage = 0.05, n.minobsinnode = 10,
                         bag.fraction = 0.8, verbose = FALSE)
p_gb <- predict(gb_final, newdata = as.matrix(xte), n.trees = gb_choice$n.trees)


pred_list <- stats::setNames(
  list(
    as.numeric(test$CO_Current),
    as.numeric(p_lm),
    as.numeric(p_knn),
    as.numeric(p_svm),
    as.numeric(p_rf),
    as.numeric(p_gb)
  ),
  c(
    "Persistence baseline (current CO)",
    "Multiple Linear Regression",
    paste0("KNN Regression (k=", best_k, ")"),
    paste0("Support Vector Regression (linear, cost=", best_cost, ")"),
    paste0("Random Forest (mtry=", best_mtry, ")"),
    paste0("Gradient Boosting (trees=", gb_choice$n.trees,
           ", depth=", gb_choice$interaction.depth, ")")
  )
)

test_metrics <- dplyr::bind_rows(lapply(names(pred_list), function(nm) metric_row(nm, yte, pred_list[[nm]]))) |>
  dplyr::arrange(RMSE_mg_m3)
write.csv(test_metrics, "outputs/test_metrics.csv", row.names = FALSE)

predictions <- data.frame(
  Timestamp = test$Timestamp,
  Actual_CO_Next_Hour = yte
)
for (nm in names(pred_list)) predictions[[make.names(nm)]] <- pred_list[[nm]]
write.csv(predictions, "outputs/test_predictions.csv", row.names = FALSE)

# Novelty / error audit: test error in higher-CO observations.
best_name <- test_metrics$Model[1]
best_col <- make.names(best_name)
best_prediction <- predictions[[best_col]]
threshold <- as.numeric(quantile(ytv, 0.75, na.rm = TRUE))
high <- yte >= threshold
audit <- data.frame(
  Segment = c("All test observations", "Higher-CO test observations (>= train+validation 75th percentile)"),
  N = c(length(yte), sum(high)),
  MAE_mg_m3 = c(mae(yte, best_prediction), if (sum(high)>0) mae(yte[high], best_prediction[high]) else NA_real_),
  RMSE_mg_m3 = c(rmse(yte, best_prediction), if (sum(high)>0) rmse(yte[high], best_prediction[high]) else NA_real_)
)
write.csv(audit, "outputs/high_pollution_error_audit.csv", row.names = FALSE)

# 4. EDA and evaluation figures.
p1 <- ggplot2::ggplot(data, ggplot2::aes(Timestamp, .data[[target]])) +
  ggplot2::geom_line(linewidth = 0.25, colour = "#176B87") +
  ggplot2::labs(title = "Hourly CO concentration over time", x = "Timestamp", y = "CO(GT) (mg/m³)") +
  ggplot2::theme_minimal()
ggplot2::ggsave("outputs/figures/co_time_series.png", p1, width = 9, height = 4.5, dpi = 160)

p2 <- ggplot2::ggplot(data, ggplot2::aes(.data[[target]])) +
  ggplot2::geom_histogram(bins = 35, fill = "#176B87", colour = "white") +
  ggplot2::labs(
    title = "Distribution of Next-Hour CO Values",
    x = "Next-Hour CO (mg/m³)",
    y = "Frequency"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave("outputs/figures/co_distribution.png", p2, width = 7, height = 4.5, dpi = 160)

p3 <- ggplot2::ggplot(test_metrics, ggplot2::aes(x = reorder(Model, RMSE_mg_m3), y = RMSE_mg_m3)) +
  ggplot2::geom_col(fill = "#176B87") + ggplot2::coord_flip() +
  ggplot2::labs(title = "Test-set model comparison", x = NULL, y = "RMSE (mg/m³)") +
  ggplot2::theme_minimal()
ggplot2::ggsave("outputs/figures/test_model_rmse.png", p3, width = 8, height = 5, dpi = 160)

p4 <- ggplot2::ggplot(data.frame(actual = yte, predicted = best_prediction),
                      ggplot2::aes(actual, predicted)) +
  ggplot2::geom_point(alpha = 0.45, colour = "#176B87") +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "#B23A48") +
  ggplot2::labs(title = paste("Actual vs predicted:", best_name),
                x = "Actual Next-Hour CO (mg/m³)",
                y = "Predicted Next-Hour CO (mg/m³)") +
  ggplot2::theme_minimal()
ggplot2::ggsave("outputs/figures/actual_vs_predicted.png", p4, width = 6.5, height = 6, dpi = 160)

png("outputs/figures/random_forest_importance.png", width = 1100, height = 750, res = 140)
randomForest::varImpPlot(rf_final, main = "Random Forest predictor importance")
dev.off()

cat("\n================ PROJECT SUMMARY ================\n")
cat("Rows with valid CO target:", nrow(data), "\n")
cat("Chronological split sizes:", length(i_train), length(i_val), nrow(test), "\n")
cat("Best model by held-out test RMSE:", test_metrics$Model[1], "\n")
print(test_metrics)
cat("\nHigh-pollution error audit:\n"); print(audit)
cat("\nOutputs saved in outputs/ and outputs/figures/.\n")
