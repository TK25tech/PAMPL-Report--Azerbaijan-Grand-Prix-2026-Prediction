# Predicting Formula 1 Finishing Positions — 2026 Azerbaijan Grand Prix

## Project Overview

This project develops a machine learning approach to predict Formula 1 driver finishing positions for the **2026 Azerbaijan Grand Prix in Baku**.

Historical Formula 1 race data from **2018–2025** is used to identify patterns in driver and constructor performance, circuit history, championship position, and weather conditions. The project deliberately excludes qualifying, practice, starting-grid and other information that would not be available within the defined prediction scenario.

## Objectives

* Explore historical Formula 1 race performance.
* Engineer driver, constructor, circuit and weather features.
* Compare regression-based machine learning models.
* Evaluate model performance using a separate validation and test period.
* Generate finishing-position predictions for the 2026 Azerbaijan Grand Prix.

## Data

The dataset contains race-level observations for individual drivers and includes variables covering:

* Driver and constructor performance
* Recent race form
* Championship standings
* Previous circuit performance
* Wet-weather performance
* Weather conditions
* Actual finishing position

The modelling data is divided chronologically to reduce information leakage:

| Dataset    | Period    | Purpose                    |
| ---------- | --------- | -------------------------- |
| Training   | 2018–2023 | Model development          |
| Validation | 2024      | Model selection and tuning |
| Test       | 2025      | Final evaluation           |
| Forecast   | 2026      | Azerbaijan GP prediction   |

## Methods

The workflow includes:

1. Data cleaning and exploratory analysis
2. Feature engineering
3. Missing-value handling and preprocessing
4. Feature scaling and transformation
5. Regression model development
6. Model validation and comparison
7. Final 2025 test evaluation
8. 2026 Azerbaijan GP forecasting

The main models evaluated include **Ridge Regression** and **Random Forest Regression**.

## Project Structure

```text
F1-Baku-2026-ML/
│
├── code/
│   └── F1_Baku_2026.py
├── data/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── report/
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* FastF1
* OpenPyXL
* Joblib

## Results

The final model is evaluated using **RMSE, MAE and R²**, with additional analysis of finishing-position and winner-prediction performance. The resulting model is then applied to the available 2026 Azerbaijan Grand Prix information to produce a forecast of driver finishing positions.

## Author

**Trisha Kampani**
MSc Data Analytics — 2026
BSBI, Berlin Campus
