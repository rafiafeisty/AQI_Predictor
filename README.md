# Global Air Quality Analysis & AQI Prediction Dashboard

## Overview

This is a comprehensive, interactive **Streamlit dashboard** for analyzing global air quality data. It allows users to:

- Upload and explore air quality datasets
- Calculate the Air Quality Index (AQI) using the official **US EPA standard**
- Perform in-depth Exploratory Data Analysis (EDA)
- Train and compare machine learning models to predict AQI values and categories
- Visualize feature importance, geographic, and seasonal patterns
- Use a live prediction tool for real-time AQI estimation

Built with **Streamlit**, **Pandas**, **Plotly**, **Matplotlib**, **Seaborn**, **Scikit-learn**, and **XGBoost**.

## Dataset Requirements

The dashboard expects a CSV file (e.g., `global_air_quality_data_10000.csv`) with at least the following columns:

| Column        | Description                       | Units / Notes                     |
|---------------|-----------------------------------|-----------------------------------|
| `Date`        | Measurement date                  | YYYY-MM-DD format                 |
| `City`        | City name                         | String                            |
| `Country`     | Country name                      | String                            |
| `PM2.5`       | Particulate Matter ≤ 2.5 µm        | µg/m³                             |
| `PM10`        | Particulate Matter ≤ 10 µm         | µg/m³                             |
| `NO2`         | Nitrogen Dioxide                  | ppb                               |
| `SO2`         | Sulfur Dioxide                    | ppb                               |
| `CO`          | Carbon Monoxide                   | mg/m³                             |
| `O3`          | Ozone                             | ppb                               |
| `Temperature` | Air temperature                   | °C                                |
| `Humidity`    | Relative humidity                 | %                                 |
| `Wind Speed`  | Wind speed                        | m/s or km/h                       |

The code works with any dataset size as long as the required columns are present.

## Dashboard Sections

1. **Data Upload & Overview**  
   Upload CSV and view shape, columns, missing values, and duplicates.

2. **Data Cleaning & Preprocessing**  
   Date parsing, Month/Day extraction, skewness analysis, and outlier detection (Z-score > 3).

3. **Exploratory Data Analysis (EDA)**  
   Interactive tabs: distributions (histograms, box, violin plots), counts by month/city/country, averages, correlation heatmaps, and seasonal trends.

4. **AQI Calculation**  
   Computes sub-indices for each pollutant using US EPA breakpoints, determines overall AQI (max sub-index), and assigns AQI category.

5. **Regression Models for AQI Prediction**  
   Predicts numerical AQI using Random Forest and XGBoost. Displays R², RMSE, MAE, category confusion matrix, and feature importance.

6. **Classification Models for AQI Category**  
   Predicts AQI category using Logistic Regression, Random Forest, XGBoost, and KNN. Shows classification reports, confusion matrices, and feature importance.

7. **Feature Importance Summary**  
   In-depth analysis of AQI drivers: correlations, geographic/seasonal patterns, Random Forest & permutation importance, partial dependence, feature interactions, and key insights.

8. **Prediction Tool**  
   Live AQI prediction using the trained Random Forest model. Users manually input pollutant and meteorological values.

## Installation & Running the App

### Requirements

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn xgboost
