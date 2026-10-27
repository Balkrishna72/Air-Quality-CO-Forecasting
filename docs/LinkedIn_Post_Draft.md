# LinkedIn post draft — publish after running the R pipeline

I’m working on an academic Predictive Analytics project: **Air Quality Prediction using R**.

Using the UCI Air Quality dataset, I built a time-aware regression workflow to estimate hourly carbon monoxide concentration from sensor-response and weather variables. I compared five model families—Multiple Linear Regression, KNN Regression, Support Vector Regression, Random Forest, and Gradient Boosting—using MAE, RMSE and R² on a later held-out time period.

A key design choice was to treat `-200` as missing, learn imputation from training data only, and keep the test period separate. I also examined errors on higher-CO observations rather than reporting a single score alone.

**Finding from my final R run:** [insert best model and metrics from `outputs/test_metrics.csv` after running the project].

This is a same-hour estimation experiment, not a future pollution forecast or a regulatory air-quality measurement.

Project repository: [add your GitHub URL]

#RStats #PredictiveAnalytics #MachineLearning #DataScience #AirQuality
