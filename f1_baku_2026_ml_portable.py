"""Predicting 2026 Azerbaijan GP finishing positions with historical race data.

Run this file from top to bottom in PyCharm. The script intentionally excludes
qualifying, starting-grid and practice information.

Required packages:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl joblib fastf1
"""

# %% PART 0 — PROJECT AND ENVIRONMENT SETUP

from pathlib import Path
import platform
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")  # Avoid Windows/Tkinter crashes; figures are saved to disk.

import joblib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# The project folder is the folder containing this Python script.
# This makes the script portable across different computers.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
WORKBOOK_PATH = DATA_DIR / "F1_Baku_2026_ML_Project_Pack.xlsx"

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
CACHE_DIR = PROJECT_DIR / "fastf1_cache"

for folder in [DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
F1_RED = "#D71920"
F1_BLUE = "#4C78A8"
F1_ORANGE = "#E68619"
F1_GREEN = "#2A936F"


def heading(title):
    print(f"\n{title}\n{'=' * 55}")


def save_figure(filename):
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


heading("PART 0 — PROJECT AND ENVIRONMENT SETUP")
print(f"Python version:       {platform.python_version()}")
print(f"Python executable:    {sys.executable}")
print(f"pandas version:       {pd.__version__}")
print(f"NumPy version:        {np.__version__}")
print(f"Matplotlib version:   {matplotlib.__version__}")
print(f"Seaborn version:      {sns.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"SciPy version:        {scipy.__version__}")
print(f"openpyxl version:     {openpyxl.__version__}")
print("\nFile checks\n" + "-" * 55)
print(f"Project folder found: {PROJECT_DIR.exists()}")
print(f"Data folder found:    {DATA_DIR.exists()}")
print(f"Workbook found:       {WORKBOOK_PATH.exists()}")
print("\nOutput folders\n" + "-" * 55)
print(f"Figures: {FIGURES_DIR}\nTables:  {TABLES_DIR}\nModels:  {MODELS_DIR}")

if not WORKBOOK_PATH.exists():
    raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

print("\nPart 0 completed successfully.")


# %% PART 1 — DATA LOADING AND INITIAL UNDERSTANDING

heading("PART 1 — DATA LOADING AND INITIAL UNDERSTANDING")

raw_data = pd.read_excel(WORKBOOK_PATH, sheet_name="Dataset")

race_keys = ["season", "round"]
season_summary = raw_data.groupby("season", as_index=False).agg(
    races=("round", "nunique"),
    rows=("driver_id", "size"),
    drivers=("driver_id", "nunique"),
    constructors=("constructor_id", "nunique"),
)

duplicate_driver_races = raw_data.duplicated(race_keys + ["driver_id"]).sum()
winner_counts = raw_data.groupby(race_keys)["win"].sum()

column_summary = pd.DataFrame({
    "column": raw_data.columns,
    "data_type": raw_data.dtypes.astype(str).values,
    "non_null": raw_data.notna().sum().values,
    "missing": raw_data.isna().sum().values,
    "unique_values": raw_data.nunique(dropna=True).values,
})

initial_quality = pd.DataFrame({
    "check": [
        "Exact duplicate rows",
        "Duplicate driver-race records",
        "Races without exactly one winner",
        "Columns containing missing values",
        "Total missing cells",
    ],
    "number_affected": [
        raw_data.duplicated().sum(),
        duplicate_driver_races,
        winner_counts.ne(1).sum(),
        raw_data.isna().any().sum(),
        raw_data.isna().sum().sum(),
    ],
})

column_summary.to_csv(TABLES_DIR / "part1_column_summary.csv", index=False)
season_summary.to_csv(TABLES_DIR / "part1_season_summary.csv", index=False)
initial_quality.to_csv(TABLES_DIR / "part1_initial_quality_summary.csv", index=False)

print(f"Dataset shape: {raw_data.shape}")
print(f"Number of seasons: {raw_data['season'].nunique()}")
print(f"Number of races: {raw_data.groupby(race_keys).ngroups}")
print(f"Number of drivers: {raw_data['driver_id'].nunique()}")
print(f"Number of constructors: {raw_data['constructor_id'].nunique()}")
print(f"Number of circuits: {raw_data['circuit_id'].nunique()}")
print("\nColumn names\n" + "-" * 55)
for number, column in enumerate(raw_data.columns, start=1):
    print(f"{number:02d}. {column}")
print("\nFirst three rows\n" + "-" * 55)
report_columns = [
    "season", "round", "grand_prix", "driver_name", "constructor",
    "driver_avg_finish_last5", "driver_points_last5",
    "driver_championship_rank_before", "weather_air_temp_c",
    "weather_rainfall", "actual_finish_position"
]
print(raw_data[report_columns].head(7).to_string(index=False))
print("\n" + "." * 110)
print("...")
print("." * 110 + "\n")
print(raw_data[report_columns].tail(2).to_string(index=False))
print("\nData types and memory information\n" + "-" * 55)
raw_data.info(memory_usage="deep")
print("\nRows and races by season\n" + "-" * 55)
print(season_summary.to_string(index=False))
print("\nInitial data-quality summary\n" + "-" * 55)
print(initial_quality.to_string(index=False))
print("\nPart 1 completed successfully.")


# %% PART 2 — DATA CLEANING AND VALIDATION

heading("PART 2 — DATA CLEANING AND VALIDATION")

cleaned_data = raw_data.copy()
rows_before = len(cleaned_data)
cleaned_data["race_date"] = pd.to_datetime(cleaned_data["race_date"], errors="coerce")

# The workbook's original constructor rank was repeated per driver. Recalculate
# one dense rank per constructor within each race from pre-race season points.
team_rank = (
    cleaned_data[["season", "round", "constructor_id", "constructor_season_points_before"]]
    .drop_duplicates(["season", "round", "constructor_id"])
)
team_rank["correct_rank"] = team_rank.groupby(race_keys)[
    "constructor_season_points_before"
].rank(method="dense", ascending=False).astype(int)

old_constructor_rank = cleaned_data["constructor_championship_rank_before"].copy()
cleaned_data = cleaned_data.drop(columns="constructor_championship_rank_before").merge(
    team_rank[race_keys + ["constructor_id", "correct_rank"]],
    on=race_keys + ["constructor_id"],
    how="left",
    validate="many_to_one",
).rename(columns={"correct_rank": "constructor_championship_rank_before"})
constructor_rank_changes = old_constructor_rank.ne(
    cleaned_data["constructor_championship_rank_before"]
).sum()

required_identifiers = [
    "season", "round", "race_date", "grand_prix", "circuit_id",
    "driver_id", "constructor_id",
]
binary_columns = ["weather_rainfall", "win", "podium", "dnf"]
field_size = cleaned_data.groupby(race_keys)["driver_id"].transform("size")
invalid_positions = (
    cleaned_data["actual_finish_position"].lt(1)
    | cleaned_data["actual_finish_position"].gt(field_size)
)
weather_outside_range = (
    ~cleaned_data["weather_air_temp_c"].between(-20, 60)
    | ~cleaned_data["weather_track_temp_c"].between(-20, 80)
    | ~cleaned_data["weather_humidity_pct"].between(0, 100)
    | ~cleaned_data["weather_wind_speed_mps"].between(0, 30)
)
historical_features = [
    column for column in cleaned_data.columns
    if "last5" in column or "at_circuit" in column or "in_wet_before" in column
]

checks = [
    ("Exact duplicate rows", cleaned_data.duplicated().sum(), "No action required",
     "Identical rows provide no additional information."),
    ("Duplicate driver-race records", cleaned_data.duplicated(race_keys + ["driver_id"]).sum(),
     "No action required", "Each driver should appear once in each race."),
    ("Invalid race dates", cleaned_data["race_date"].isna().sum(),
     "Converted race_date to datetime", "Dates are needed for chronological ordering."),
    ("Missing required identifiers", cleaned_data[required_identifiers].isna().any(axis=1).sum(),
     "No action required", "Identifiers are required for grouping and labelling."),
    ("Invalid finishing positions", invalid_positions.sum(), "No action required",
     "A finishing position must fall within the race field size."),
    ("Duplicate positions within a race",
     cleaned_data.duplicated(race_keys + ["actual_finish_position"]).sum(), "No action required",
     "The ranking target should contain one unique position per driver."),
    ("Invalid binary values", sum((~cleaned_data[c].isin([0, 1])).sum() for c in binary_columns),
     "No action required", "Binary fields should contain only 0 or 1."),
    ("Races without exactly one winner", cleaned_data.groupby(race_keys)["win"].sum().ne(1).sum(),
     "No action required", "Every race should have exactly one winner."),
    ("Winner-position inconsistencies",
     (cleaned_data["win"].eq(1) != cleaned_data["actual_finish_position"].eq(1)).sum(),
     "No action required", "The winner should have finishing position 1."),
    ("Podium-position inconsistencies",
     (cleaned_data["podium"].eq(1) != cleaned_data["actual_finish_position"].le(3)).sum(),
     "No action required", "Podium should correspond to positions 1–3."),
    ("Missing weather values",
     cleaned_data[[c for c in cleaned_data if c.startswith("weather_") and c != "weather_source"]]
     .isna().any(axis=1).sum(), "No action required",
     "Weather is required for the dry and wet scenarios."),
    ("Weather values outside plausible ranges", weather_outside_range.sum(), "No action required",
     "Weather values should remain within broad physical limits."),
    ("Incorrect constructor championship ranks", constructor_rank_changes,
     "Recalculated dense rank per constructor and race",
     "Constructor rank must represent 10 teams historically, not driver rows."),
    ("Rows with missing historical features",
     cleaned_data[historical_features].isna().any(axis=1).sum(), "Retained",
     "Early-career and first-circuit records are valid observations."),
]

cleaning_summary = pd.DataFrame(checks, columns=[
    "issue_checked", "number_affected", "action_taken", "justification"
])
cleaning_summary["rows_before"] = rows_before
cleaning_summary["rows_after"] = len(cleaned_data)

cleaning_summary.to_csv(TABLES_DIR / "part2_cleaning_summary.csv", index=False)
cleaned_data.to_csv(TABLES_DIR / "part2_cleaned_dataset.csv", index=False)

print("Cleaning and validation summary\n" + "-" * 55)
print(cleaning_summary.to_string(index=False))
print(f"\nRows before cleaning: {rows_before:,}")
print(f"Rows after cleaning:  {len(cleaned_data):,}")
print(f"race_date type:       {cleaned_data['race_date'].dtype}")
print(f"Maximum constructor rank: {cleaned_data['constructor_championship_rank_before'].max()}")
print("\nPart 2 completed successfully.")


# %% PART 3 — DATA TYPES AND DATA DICTIONARY

heading("PART 3 — DATA TYPES AND DATA DICTIONARY")

TARGET = "actual_finish_position"
NUMERICAL_FEATURES = [
    "driver_prior_starts", "driver_avg_finish_last5", "driver_points_last5",
    "driver_wins_last5", "driver_podiums_last5", "driver_dnf_rate_last5",
    "driver_season_points_before", "driver_season_wins_before",
    "driver_circuit_starts_before", "driver_avg_finish_at_circuit",
    "driver_wins_at_circuit", "driver_wet_starts_before",
    "driver_avg_finish_in_wet_before", "constructor_points_last5",
    "constructor_wins_last5", "constructor_dnf_rate_last5",
    "constructor_season_points_before", "constructor_season_wins_before",
    "weather_air_temp_c", "weather_track_temp_c", "weather_humidity_pct",
    "weather_wind_speed_mps",
]
ORDINAL_FEATURES = [
    "driver_championship_rank_before", "constructor_championship_rank_before"
]
CATEGORICAL_FEATURES = ["circuit_id", "driver_id", "constructor_id"]
BOOLEAN_FEATURES = ["weather_rainfall"]
APPROVED_FEATURES = (
    NUMERICAL_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
)

IDENTIFIER_COLUMNS = [
    "season", "round", "race_date", "grand_prix", "circuit_name",
    "circuit_location", "country", "driver_code", "driver_name", "constructor",
]
LEAKAGE_COLUMNS = [
    "win", "classified_position", "points_scored", "status", "podium", "dnf",
    "weather_observations", "weather_source",
]


def variable_type(column):
    if column == TARGET:
        return "Ordinal", "Target", "Target only"
    if column in LEAKAGE_COLUMNS:
        kind = "Boolean" if column in ["win", "podium", "dnf"] else (
            "Numerical" if column in ["points_scored", "weather_observations"] else "Categorical"
        )
        return kind, "Outcome/leakage", "No"
    if column in IDENTIFIER_COLUMNS:
        kind = "Date/time" if column == "race_date" else (
            "Numerical" if column == "season" else (
                "Ordinal" if column == "round" else "Categorical"
            )
        )
        return kind, "Identifier/display", "No"
    if column in NUMERICAL_FEATURES:
        return "Numerical", "Predictor", "Yes"
    if column in ORDINAL_FEATURES:
        return "Ordinal", "Predictor", "Yes"
    if column in CATEGORICAL_FEATURES:
        return "Categorical", "Predictor", "Yes"
    if column in BOOLEAN_FEATURES:
        return "Boolean", "Predictor", "Yes"
    return "Unclassified", "Review", "No"


dictionary_rows = []
for column in cleaned_data.columns:
    data_type, role, use = variable_type(column)
    dictionary_rows.append({
        "column": column, "data_type": data_type, "role": role, "use_in_model": use,
        "missing_values": cleaned_data[column].isna().sum(),
        "example": cleaned_data[column].dropna().iloc[0] if cleaned_data[column].notna().any() else np.nan,
    })

data_dictionary = pd.DataFrame(dictionary_rows)
data_dictionary.to_csv(TABLES_DIR / "part3_data_dictionary.csv", index=False)

print("Variables by role\n" + "-" * 55)
print(data_dictionary["role"].value_counts().to_string())
print("\nApproved predictor counts\n" + "-" * 55)
print(f"Numerical predictors:   {len(NUMERICAL_FEATURES)}")
print(f"Ordinal predictors:     {len(ORDINAL_FEATURES)}")
print(f"Categorical predictors: {len(CATEGORICAL_FEATURES)}")
print(f"Boolean predictors:     {len(BOOLEAN_FEATURES)}")
print(f"Total predictors:       {len(APPROVED_FEATURES)}")
print("\nData dictionary\n" + "-" * 55)
print(data_dictionary[["column", "data_type", "role", "use_in_model"]].to_string(index=False))
print("\nPart 3 completed successfully.")


# %% PART 4 — EXPLORATORY DATA ANALYSIS

heading("PART 4 — EXPLORATORY DATA ANALYSIS")

eda_columns = NUMERICAL_FEATURES + ORDINAL_FEATURES + [TARGET]
descriptive_statistics = cleaned_data[eda_columns].describe().T
descriptive_statistics["median"] = cleaned_data[eda_columns].median()
descriptive_statistics["missing_values"] = cleaned_data[eda_columns].isna().sum()
descriptive_statistics = descriptive_statistics[
    ["count", "mean", "median", "std", "min", "max", "missing_values"]
]
descriptive_statistics.to_csv(TABLES_DIR / "part4_descriptive_statistics.csv")


def scatter_with_rho(x, y, title, xlabel, color, filename):
    subset = cleaned_data[[x, y]].dropna()
    rho = spearmanr(subset[x], subset[y]).statistic
    plt.figure(figsize=(10, 7))
    sns.regplot(data=subset, x=x, y=y, scatter_kws={"alpha": 0.28, "s": 30},
                line_kws={"color": "#111111"}, color=color)
    plt.title(f"{title}\nSpearman correlation = {rho:.2f}", weight="bold")
    plt.xlabel(xlabel)
    plt.ylabel("Actual finishing position")
    save_figure(filename)


plt.figure(figsize=(10, 6))
sns.countplot(data=cleaned_data, x=TARGET, color=F1_RED)
plt.title("Distribution of Actual Finishing Positions", weight="bold")
plt.xlabel("Actual finishing position")
plt.ylabel("Number of driver-race records")
save_figure("part4_figure1_finish_distribution.png")

scatter_with_rho(
    "driver_avg_finish_last5", TARGET,
    "Recent Driver Form and Race Finishing Position",
    "Average finishing position in previous five races", F1_RED,
    "part4_figure2_driver_recent_form.png",
)
scatter_with_rho(
    "constructor_points_last5", TARGET,
    "Recent Constructor Form and Race Finishing Position",
    "Constructor points in previous five races", F1_BLUE,
    "part4_figure3_constructor_form.png",
)
scatter_with_rho(
    "driver_championship_rank_before", TARGET,
    "Pre-Race Championship Rank and Finishing Position",
    "Championship rank before the race", F1_ORANGE,
    "part4_figure4_championship_rank.png",
)
scatter_with_rho(
    "driver_dnf_rate_last5", TARGET,
    "Recent DNF Rate and Race Finishing Position",
    "Driver DNF rate in previous five races", F1_GREEN,
    "part4_figure5_dnf_history.png",
)

weather_plot = cleaned_data[[
    "driver_avg_finish_in_wet_before", TARGET, "weather_rainfall"
]].dropna()
plt.figure(figsize=(10, 7))
for rainfall, label, colour in [(0, "Dry race", F1_BLUE), (1, "Wet race", F1_RED)]:
    subset = weather_plot[weather_plot["weather_rainfall"] == rainfall]
    sns.regplot(data=subset, x="driver_avg_finish_in_wet_before", y=TARGET,
                scatter_kws={"alpha": 0.25, "s": 28}, color=colour, label=label)
plt.title("Prior Wet-Race Performance Under Dry and Wet Conditions", weight="bold")
plt.xlabel("Average finishing position in previous wet races")
plt.ylabel("Actual finishing position")
plt.legend()
save_figure("part4_figure6_weather_relationship.png")

correlation_columns = {
    "driver_avg_finish_last5": "Driver recent finish",
    "driver_points_last5": "Driver recent points",
    "driver_dnf_rate_last5": "Driver DNF rate",
    "driver_championship_rank_before": "Driver rank",
    "driver_circuit_starts_before": "Circuit starts",
    "driver_avg_finish_at_circuit": "Circuit avg finish",
    "constructor_points_last5": "Constructor points",
    "constructor_dnf_rate_last5": "Constructor DNF rate",
    "constructor_championship_rank_before": "Constructor rank",
    "weather_rainfall": "Rainfall",
    TARGET: "Actual finish",
}
correlation_matrix = cleaned_data[list(correlation_columns)].corr(method="spearman")
correlation_matrix = correlation_matrix.rename(index=correlation_columns, columns=correlation_columns)
correlation_matrix.to_csv(TABLES_DIR / "part4_correlation_matrix.csv")
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Spearman Correlations Between Selected Variables", weight="bold")
save_figure("part4_figure7_correlation_heatmap.png")

baku_data = cleaned_data[cleaned_data["circuit_id"] == "baku"]
baku_summary = baku_data.groupby("driver_name", as_index=False).agg(
    starts=(TARGET, "size"), average_finish=(TARGET, "mean")
)
baku_summary = baku_summary[baku_summary["starts"] >= 2].sort_values("average_finish").head(12)
baku_summary.to_csv(TABLES_DIR / "part4_baku_driver_summary.csv", index=False)
plt.figure(figsize=(11, 8))
labels = baku_summary["driver_name"] + " (" + baku_summary["starts"].astype(str) + " starts)"
bars = plt.barh(labels, baku_summary["average_finish"], color=F1_RED)
plt.gca().invert_yaxis()
plt.bar_label(bars, fmt="%.1f", padding=4)
plt.title("Best Historical Average Finishes at Baku\nMinimum Two Starts", weight="bold")
plt.xlabel("Average finishing position — lower is better")
plt.ylabel("Driver")
save_figure("part4_figure8_baku_history.png")

print("Descriptive statistics\n" + "-" * 55)
print(descriptive_statistics.round(3).to_string())
print(f"\nBaku historical races: {baku_data.groupby(race_keys).ngroups}")
print("Figures saved: 8")
print("\nPart 4 completed successfully.")


# %% PART 5 — FEATURE SELECTION AND PREPROCESSING

heading("PART 5 — FEATURE SELECTION AND PREPROCESSING")

train_data = cleaned_data[cleaned_data["season"].between(2018, 2023)].copy()
validation_data = cleaned_data[cleaned_data["season"] == 2024].copy()
test_data = cleaned_data[cleaned_data["season"] == 2025].copy()

X_train, y_train = train_data[APPROVED_FEATURES], train_data[TARGET]
X_validation, y_validation = validation_data[APPROVED_FEATURES], validation_data[TARGET]
X_test, y_test = test_data[APPROVED_FEATURES], test_data[TARGET]

forbidden_columns = [TARGET] + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS
forbidden_in_X = sorted(set(APPROVED_FEATURES).intersection(forbidden_columns))

scaled_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])
unscaled_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
boolean_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

ridge_preprocessor = ColumnTransformer([
    ("numeric", scaled_numeric, NUMERICAL_FEATURES + ORDINAL_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ("boolean", boolean_pipeline, BOOLEAN_FEATURES),
])
tree_preprocessor = ColumnTransformer([
    ("numeric", unscaled_numeric, NUMERICAL_FEATURES + ORDINAL_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ("boolean", boolean_pipeline, BOOLEAN_FEATURES),
])

preprocessing_check = clone(ridge_preprocessor)
train_matrix = preprocessing_check.fit_transform(X_train)
validation_matrix = preprocessing_check.transform(X_validation)
test_matrix = preprocessing_check.transform(X_test)

split_summary = pd.DataFrame([
    ["Training", "2018–2023", len(train_data), train_data.groupby(race_keys).ngroups],
    ["Validation", "2024", len(validation_data), validation_data.groupby(race_keys).ngroups],
    ["Final test", "2025", len(test_data), test_data.groupby(race_keys).ngroups],
], columns=["period", "seasons", "rows", "races"])

feature_group_summary = pd.DataFrame([
    ["Numerical", len(NUMERICAL_FEATURES), "Median imputation; Ridge scaling"],
    ["Ordinal", len(ORDINAL_FEATURES), "Median imputation; Ridge scaling"],
    ["Categorical", len(CATEGORICAL_FEATURES), "Most-frequent imputation and one-hot encoding"],
    ["Boolean", len(BOOLEAN_FEATURES), "Most-frequent imputation; retain as 0/1"],
], columns=["feature_group", "number_of_features", "treatment"])

split_summary.to_csv(TABLES_DIR / "part5_chronological_split.csv", index=False)
feature_group_summary.to_csv(TABLES_DIR / "part5_feature_groups.csv", index=False)

print("Chronological split\n" + "-" * 55)
print(split_summary.to_string(index=False))
print("\nFeature groups\n" + "-" * 55)
print(feature_group_summary.to_string(index=False))
print("\nPreprocessing checks\n" + "-" * 55)
print(f"Target: {TARGET}")
print(f"Approved raw predictors: {len(APPROVED_FEATURES)}")
print(f"Forbidden columns in X: {forbidden_in_X}")
print(f"Training matrix shape:   {train_matrix.shape}")
print(f"Validation matrix shape: {validation_matrix.shape}")
print(f"Test matrix shape:       {test_matrix.shape}")
print("\nPart 5 completed successfully.")


# %% PART 6 — BASELINE AND RIDGE REGRESSION

heading("PART 6 — BASELINE AND RIDGE REGRESSION")


def metric_row(model_name, dataset_name, actual, predicted, runtime=0.0):
    predicted = np.asarray(predicted)
    rho = np.nan if np.unique(predicted).size == 1 else spearmanr(actual, predicted).statistic
    return {
        "model": model_name,
        "dataset": dataset_name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R_squared": r2_score(actual, predicted),
        "Spearman": rho,
        "runtime_seconds": runtime,
    }


training_median = y_train.median()
baseline_train_prediction = np.full(len(y_train), training_median)
baseline_validation_prediction = np.full(len(y_validation), training_median)

ridge_model = Pipeline([
    ("preprocessor", clone(ridge_preprocessor)),
    ("model", Ridge(alpha=1.0)),
])
start_time = time.perf_counter()
ridge_model.fit(X_train, y_train)
ridge_runtime = time.perf_counter() - start_time
ridge_train_prediction = ridge_model.predict(X_train)
ridge_validation_prediction = ridge_model.predict(X_validation)

part6_results = pd.DataFrame([
    metric_row("Median baseline", "Training", y_train, baseline_train_prediction),
    metric_row("Median baseline", "Validation", y_validation, baseline_validation_prediction),
    metric_row("Ridge Regression", "Training", y_train, ridge_train_prediction, ridge_runtime),
    metric_row("Ridge Regression", "Validation", y_validation, ridge_validation_prediction, ridge_runtime),
])
part6_results.to_csv(TABLES_DIR / "part6_baseline_ridge_results.csv", index=False)
joblib.dump(ridge_model, MODELS_DIR / "part6_ridge_model.joblib")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].scatter(y_validation, ridge_validation_prediction, alpha=0.5, color=F1_BLUE)
axes[0].plot([1, 20], [1, 20], "r--")
axes[0].set(title="Ridge: Actual vs Predicted", xlabel="Actual finishing position",
            ylabel="Predicted finishing-position score")
residuals = y_validation.to_numpy() - ridge_validation_prediction
axes[1].scatter(ridge_validation_prediction, residuals, alpha=0.5, color=F1_ORANGE)
axes[1].axhline(0, color="#111111", linestyle="--")
axes[1].set(title="Ridge: Validation Residuals", xlabel="Predicted finishing-position score",
            ylabel="Residual: actual minus predicted")
plt.tight_layout()
save_figure("part6_ridge_diagnostics.png")

print(f"Training median used by baseline: {training_median:.2f}")
print("\nBaseline and Ridge results\n" + "-" * 55)
print(part6_results.round(3).to_string(index=False))
print("\nPart 6 completed successfully.")
print("The untouched 2025 test set was not evaluated.")


# %% PART 7 — DECISION TREE REGRESSOR

heading("PART 7 — DECISION TREE REGRESSOR")


def raw_feature_importance(fitted_pipeline):
    transformed_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = fitted_pipeline.named_steps["model"].feature_importances_
    rows = []
    features_by_length = sorted(APPROVED_FEATURES, key=len, reverse=True)
    for transformed_name, importance in zip(transformed_names, importances):
        name = transformed_name.split("__", 1)[-1]
        if name.startswith("missingindicator_"):
            raw_name = "missing_value_indicators"
        else:
            raw_name = next(
                (feature for feature in features_by_length
                 if name == feature or name.startswith(feature + "_")),
                name,
            )
        rows.append((raw_name, importance))
    return (
        pd.DataFrame(rows, columns=["feature", "importance"])
        .groupby("feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
    )


def importance_chart(importance_table, title, colour, filename, top_n=15):
    chart_data = importance_table.head(top_n).sort_values("importance")
    plt.figure(figsize=(11, 8))
    plt.barh(chart_data["feature"], chart_data["importance"], color=colour)
    plt.title(title, weight="bold")
    plt.xlabel("Aggregated impurity-based importance")
    plt.ylabel("Feature")
    save_figure(filename)


decision_tree_model = Pipeline([
    ("preprocessor", clone(tree_preprocessor)),
    ("model", DecisionTreeRegressor(
        max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
    )),
])
start_time = time.perf_counter()
decision_tree_model.fit(X_train, y_train)
tree_runtime = time.perf_counter() - start_time
tree_train_prediction = decision_tree_model.predict(X_train)
tree_validation_prediction = decision_tree_model.predict(X_validation)

part7_results = pd.DataFrame([
    metric_row("Decision Tree", "Training", y_train, tree_train_prediction, tree_runtime),
    metric_row("Decision Tree", "Validation", y_validation, tree_validation_prediction, tree_runtime),
])
tree_importance = raw_feature_importance(decision_tree_model)
part7_results.to_csv(TABLES_DIR / "part7_decision_tree_results.csv", index=False)
tree_importance.to_csv(TABLES_DIR / "part7_decision_tree_importance.csv", index=False)
joblib.dump(decision_tree_model, MODELS_DIR / "part7_decision_tree_model.joblib")
importance_chart(
    tree_importance, "Decision Tree Feature Importance", F1_GREEN,
    "part7_decision_tree_importance.png",
)

fitted_tree = decision_tree_model.named_steps["model"]
tree_gap = part7_results.loc[part7_results["dataset"] == "Validation", "MAE"].iloc[0] - \
           part7_results.loc[part7_results["dataset"] == "Training", "MAE"].iloc[0]
print("Decision Tree settings\n" + "-" * 55)
print("Maximum allowed depth: 6")
print(f"Actual fitted depth:   {fitted_tree.get_depth()}")
print(f"Number of leaf nodes:  {fitted_tree.get_n_leaves()}")
print("Minimum samples/leaf:  20")
print("\nDecision Tree results\n" + "-" * 55)
print(part7_results.round(3).to_string(index=False))
print(f"\nValidation–training MAE gap: {tree_gap:.3f}")
print("\nPart 7 completed successfully.")
print("The untouched 2025 test set was not evaluated.")


# %% PART 8 — RANDOM FOREST REGRESSOR

heading("PART 8 — RANDOM FOREST REGRESSOR")

random_forest_model = Pipeline([
    ("preprocessor", clone(tree_preprocessor)),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=1,  # Stable with Python 3.14 on Windows.
    )),
])
start_time = time.perf_counter()
random_forest_model.fit(X_train, y_train)
forest_runtime = time.perf_counter() - start_time
forest_train_prediction = random_forest_model.predict(X_train)
forest_validation_prediction = random_forest_model.predict(X_validation)

part8_results = pd.DataFrame([
    metric_row("Random Forest", "Training", y_train, forest_train_prediction, forest_runtime),
    metric_row("Random Forest", "Validation", y_validation, forest_validation_prediction, forest_runtime),
])
forest_importance = raw_feature_importance(random_forest_model)
part8_results.to_csv(TABLES_DIR / "part8_random_forest_results.csv", index=False)
forest_importance.to_csv(TABLES_DIR / "part8_random_forest_importance.csv", index=False)
joblib.dump(random_forest_model, MODELS_DIR / "part8_random_forest_model.joblib")
importance_chart(
    forest_importance, "Random Forest Feature Importance", F1_BLUE,
    "part8_random_forest_importance.png",
)

forest_gap = part8_results.loc[part8_results["dataset"] == "Validation", "MAE"].iloc[0] - \
             part8_results.loc[part8_results["dataset"] == "Training", "MAE"].iloc[0]
print("Random Forest settings\n" + "-" * 55)
print("Number of trees:       300")
print("Maximum tree depth:    12")
print("Minimum samples/leaf:  5")
print("Features per split:    70%")
print(f"Training runtime:      {forest_runtime:.3f} seconds")
print("\nRandom Forest results\n" + "-" * 55)
print(part8_results.round(3).to_string(index=False))
print(f"\nValidation–training MAE gap: {forest_gap:.3f}")
print("\nPart 8 completed successfully.")
print("The untouched 2025 test set was not evaluated.")


# %% PART 9 — MODEL COMPARISON AND FINAL TEST

heading("PART 9 — MODEL COMPARISON AND FINAL TEST")


def result_value(table, model, dataset, column):
    return table.loc[
        (table["model"] == model) & (table["dataset"] == dataset), column
    ].iloc[0]


comparison_rows = []
for model_name, result_table in [
    ("Median baseline", part6_results),
    ("Ridge Regression", part6_results),
    ("Decision Tree", part7_results),
    ("Random Forest", part8_results),
]:
    train_mae = result_value(result_table, model_name, "Training", "MAE")
    validation_mae = result_value(result_table, model_name, "Validation", "MAE")
    comparison_rows.append({
        "model": model_name,
        "training_MAE": train_mae,
        "validation_MAE": validation_mae,
        "validation_RMSE": result_value(result_table, model_name, "Validation", "RMSE"),
        "validation_R_squared": result_value(result_table, model_name, "Validation", "R_squared"),
        "validation_Spearman": result_value(result_table, model_name, "Validation", "Spearman"),
        "MAE_gap": validation_mae - train_mae,
        "runtime_seconds": result_value(result_table, model_name, "Validation", "runtime_seconds"),
    })

model_comparison = pd.DataFrame(comparison_rows)
selected_model_name = "Random Forest"

# Hyperparameters were selected using validation only. Refit on 2018–2024,
# then evaluate once on the untouched 2025 final test period.
development_data = pd.concat([train_data, validation_data], ignore_index=True)
final_model = clone(random_forest_model)
final_model.fit(development_data[APPROVED_FEATURES], development_data[TARGET])
final_test_prediction = final_model.predict(X_test)
final_test_results = pd.DataFrame([
    metric_row(selected_model_name, "Final test 2025", y_test, final_test_prediction, forest_runtime)
])

model_comparison.to_csv(TABLES_DIR / "part9_model_comparison.csv", index=False)
final_test_results.to_csv(TABLES_DIR / "part9_final_test_results.csv", index=False)
joblib.dump(final_model, MODELS_DIR / "part9_selected_final_model.joblib")

chart_data = model_comparison.sort_values("validation_RMSE", ascending=False)
plt.figure(figsize=(10, 6))
bars = plt.barh(chart_data["model"], chart_data["validation_RMSE"], color=F1_RED)
plt.bar_label(bars, fmt="%.2f", padding=4)
plt.title("Validation RMSE by Model", weight="bold")
plt.xlabel("Validation RMSE — lower is better")
plt.ylabel("Model")
save_figure("part9_model_comparison.png")

print("Model comparison\n" + "-" * 55)
print(model_comparison.round(3).to_string(index=False))
print(f"\nSelected model: {selected_model_name}")
print("Selection basis: lowest validation RMSE and MAE, confirmed using R² and Spearman correlation.")
print("\nFinal untouched 2025 test result\n" + "-" * 55)
print(final_test_results.round(3).to_string(index=False))
print("\nPart 9 completed successfully.")
print("The 2025 test set will not be used for model selection.")


# %% PART 10 — RACE-LEVEL RANKING EVALUATION

heading("PART 10 — RACE-LEVEL RANKING EVALUATION")

ranking_predictions = test_data[
    ["season", "round", "grand_prix", "driver_id", "driver_name", TARGET,
     "weather_rainfall", "dnf"]
].copy()
ranking_predictions["predicted_score"] = final_test_prediction
ranking_predictions = ranking_predictions.sort_values(
    ["season", "round", "predicted_score", "driver_name"]
)
ranking_predictions["predicted_position"] = (
    ranking_predictions.groupby(race_keys).cumcount() + 1
)
ranking_predictions["absolute_position_error"] = (
    ranking_predictions["predicted_position"] - ranking_predictions[TARGET]
).abs()

race_rows = []
for (season, race_round), race in ranking_predictions.groupby(race_keys):
    predicted_winner = race.loc[race["predicted_position"].idxmin(), "driver_id"]
    actual_winner = race.loc[race[TARGET].idxmin(), "driver_id"]
    predicted_podium = set(race.nsmallest(3, "predicted_position")["driver_id"])
    actual_podium = set(race.nsmallest(3, TARGET)["driver_id"])
    predicted_top_ten = set(race.nsmallest(10, "predicted_position")["driver_id"])
    actual_top_ten = set(race.nsmallest(10, TARGET)["driver_id"])
    race_rows.append({
        "season": season,
        "round": race_round,
        "grand_prix": race["grand_prix"].iloc[0],
        "winner_correct": int(predicted_winner == actual_winner),
        "podium_overlap": len(predicted_podium & actual_podium),
        "top_ten_overlap": len(predicted_top_ten & actual_top_ten),
        "mean_absolute_rank_error": race["absolute_position_error"].mean(),
        "race_spearman": spearmanr(race[TARGET], race["predicted_position"]).statistic,
    })

race_performance = pd.DataFrame(race_rows)
winner_accuracy = race_performance["winner_correct"].mean()
podium_overlap = race_performance["podium_overlap"].mean()
top_ten_overlap = race_performance["top_ten_overlap"].mean()
rank_mae = ranking_predictions["absolute_position_error"].mean()
best_race = race_performance.loc[race_performance["mean_absolute_rank_error"].idxmin()]
worst_race = race_performance.loc[race_performance["mean_absolute_rank_error"].idxmax()]

ranking_predictions.to_csv(TABLES_DIR / "part10_driver_rank_predictions.csv", index=False)
race_performance.to_csv(TABLES_DIR / "part10_race_level_performance.csv", index=False)

plt.figure(figsize=(10, 8))
plt.scatter(ranking_predictions[TARGET], ranking_predictions["predicted_position"],
            alpha=0.35, color=F1_BLUE)
plt.plot([1, 22], [1, 22], "r--")
plt.title("Predicted vs Actual Race Rankings — 2025", weight="bold")
plt.xlabel("Actual finishing position")
plt.ylabel("Predicted finishing position")
plt.xticks(range(1, 23))
plt.yticks(range(1, 23))
save_figure("part10_predicted_vs_actual_rank.png")

error_counts = ranking_predictions["absolute_position_error"].value_counts().sort_index()
plt.figure(figsize=(11, 7))
plt.bar(error_counts.index, error_counts.values, color=F1_ORANGE)
plt.title("Distribution of Absolute Position Errors — 2025", weight="bold")
plt.xlabel("Absolute difference between predicted and actual position")
plt.ylabel("Number of driver-race predictions")
save_figure("part10_position_error_distribution.png")

print("Overall 2025 ranking performance\n" + "-" * 55)
print(f"Winner accuracy:              {winner_accuracy:.1%}")
print(f"Average podium overlap:       {podium_overlap:.2f} of 3")
print(f"Average top-ten overlap:      {top_ten_overlap:.2f} of 10")
print(f"Average absolute rank error:  {rank_mae:.2f}")
print("\nBest predicted race\n" + "-" * 55)
print(best_race[["grand_prix", "mean_absolute_rank_error", "race_spearman"]].to_string())
print("\nWorst predicted race\n" + "-" * 55)
print(worst_race[["grand_prix", "mean_absolute_rank_error", "race_spearman"]].to_string())
for label, selected_race in [("Best", best_race), ("Worst", worst_race)]:
    rows = ranking_predictions[
        ranking_predictions["round"] == selected_race["round"]
    ].nsmallest(10, "predicted_position")
    print(f"\n{label}-race predicted order\n" + "-" * 55)
    print(rows[["predicted_position", "driver_name", TARGET,
                "absolute_position_error"]].to_string(index=False))
print("\nPart 10 completed successfully.")


# %% PART 11 — INTERPRETATION AND CRITICAL ANALYSIS

heading("PART 11 — INTERPRETATION AND CRITICAL ANALYSIS")

final_importance = raw_feature_importance(final_model)


def feature_group(feature):
    if feature.startswith("constructor_") or feature == "constructor_id":
        return "Constructor"
    if feature.startswith("driver_") or feature == "driver_id":
        return "Driver"
    if feature.startswith("weather_"):
        return "Weather"
    if feature.startswith("circuit_") or feature == "circuit_id":
        return "Circuit"
    return "Other"


final_importance["feature_group"] = final_importance["feature"].map(feature_group)
group_importance = final_importance.groupby("feature_group", as_index=False)["importance"].sum()
group_importance["percentage"] = 100 * group_importance["importance"] / group_importance["importance"].sum()
group_importance = group_importance.sort_values("importance", ascending=False)

weather_rows = []
for rainfall, label in [(0, "Dry"), (1, "Wet")]:
    subset = ranking_predictions[ranking_predictions["weather_rainfall"] == rainfall]
    weather_rows.append({
        "weather": label,
        "races": subset.groupby(race_keys).ngroups,
        "driver_predictions": len(subset),
        "score_MAE": mean_absolute_error(subset[TARGET], subset["predicted_score"]),
        "rank_MAE": subset["absolute_position_error"].mean(),
        "rank_Spearman": spearmanr(subset[TARGET], subset["predicted_position"]).statistic,
    })
weather_performance = pd.DataFrame(weather_rows)

dnf_rows = []
for dnf_value, label in [(0, "No DNF"), (1, "DNF")]:
    subset = ranking_predictions[ranking_predictions["dnf"] == dnf_value]
    dnf_rows.append({
        "outcome": label,
        "driver_predictions": len(subset),
        "score_MAE": mean_absolute_error(subset[TARGET], subset["predicted_score"]),
        "rank_MAE": subset["absolute_position_error"].mean(),
    })
dnf_performance = pd.DataFrame(dnf_rows)
large_errors = ranking_predictions[ranking_predictions["absolute_position_error"] >= 10].copy()
large_error_dnf_share = large_errors["dnf"].mean() if len(large_errors) else np.nan

final_importance.to_csv(TABLES_DIR / "part11_feature_importance.csv", index=False)
group_importance.to_csv(TABLES_DIR / "part11_feature_group_importance.csv", index=False)
weather_performance.to_csv(TABLES_DIR / "part11_weather_performance.csv", index=False)
dnf_performance.to_csv(TABLES_DIR / "part11_dnf_performance.csv", index=False)
large_errors.to_csv(TABLES_DIR / "part11_large_error_cases.csv", index=False)

chart_data = group_importance.sort_values("percentage")
plt.figure(figsize=(10, 7))
bars = plt.barh(chart_data["feature_group"], chart_data["percentage"], color=F1_BLUE)
plt.bar_label(bars, fmt="%.1f%%", padding=4)
plt.title("Random Forest Importance by Feature Group", weight="bold")
plt.xlabel("Share of total impurity-based importance")
plt.ylabel("Feature group")
save_figure("part11_feature_group_importance.png")

critical_analysis = """Critical interpretation notes
- Feature importance describes how the fitted forest split the data; it does not prove causation.
- Constructor and driver history represent performance associations available before a race.
- Wet and dry subgroup results use relatively few wet races, so differences are uncertain.
- Accidents, failures, penalties and safety cars are difficult to forecast from pre-race history.
- Excluding qualifying improves pre-event usability but removes a strong signal of current pace.
- Major 2026 technical regulations may cause concept drift relative to 2018–2025 patterns.
- Forecasts should be reported as uncertain analytical scenarios, not guaranteed outcomes.
- Responsible use requires transparent sources, leakage controls and honest limitations.
"""
(TABLES_DIR / "part11_critical_analysis.txt").write_text(critical_analysis, encoding="utf-8")

print("Feature importance by group\n" + "-" * 55)
print(group_importance.round(3).to_string(index=False))
print("\nPerformance by weather\n" + "-" * 55)
print(weather_performance.round(3).to_string(index=False))
print("\nPerformance by actual DNF outcome\n" + "-" * 55)
print(dnf_performance.round(3).to_string(index=False))
print("\nLarge-error analysis\n" + "-" * 55)
print(f"Predictions wrong by 10+ positions: {len(large_errors)}")
print(f"Share involving a DNF: {large_error_dnf_share:.1%}")
print("\nPart 11 completed successfully.")
print("No model parameters or predictions were changed.")


# %% PART 12 — COMPLETED 2026 DATA UPDATE

heading("PART 12 — COMPLETED 2026 DATA UPDATE")

import fastf1
from fastf1.ergast import Ergast

fastf1.Cache.enable_cache(str(CACHE_DIR))
retrieval_time = pd.Timestamp.now(tz="Europe/Berlin")
print(f"Retrieval time: {retrieval_time}")


def load_race_weather(season, race_round, race_name):
    print(f"Loading weather: round {race_round}, {race_name}")
    session = fastf1.get_session(season, race_round, "R")
    session.load(laps=False, telemetry=False, weather=True, messages=False)
    weather = session.weather_data
    if weather is None or weather.empty:
        raise ValueError(f"No weather data found for {race_name}.")
    return {
        "weather_air_temp_c": weather["AirTemp"].mean(),
        "weather_track_temp_c": weather["TrackTemp"].mean(),
        "weather_humidity_pct": weather["Humidity"].mean(),
        "weather_wind_speed_mps": weather["WindSpeed"].mean(),
        "weather_rainfall": int(
            pd.to_numeric(weather["Rainfall"], errors="coerce").fillna(0).gt(0).any()
        ),
        "weather_observations": len(weather),
        "weather_source": f"FastF1 live timing: {session.api_path}/WeatherData.jsonStream",
    }


ergast = Ergast(result_type="pandas", auto_cast=True)
race_schedule = pd.DataFrame(ergast.get_race_schedule(season=2026, limit=100)).copy()
race_schedule["raceDate"] = pd.to_datetime(race_schedule["raceDate"])
cutoff_date = retrieval_time.tz_localize(None).normalize()
eligible_races = race_schedule[
    (race_schedule["raceDate"] <= cutoff_date)
    & ~race_schedule["raceName"].str.contains("Azerbaijan", case=False, na=False)
].reset_index(drop=True)

race_frames = []
for _, race_info in eligible_races.iterrows():
    race_round = int(race_info["round"])
    race_name = race_info["raceName"]
    result_response = ergast.get_race_results(season=2026, round=race_round, limit=100)
    if not result_response.content or result_response.content[0].empty:
        raise ValueError(f"No completed result found for round {race_round}: {race_name}")
    race_results = result_response.content[0].copy()
    if len(race_results) != 22:
        raise ValueError(f"{race_name} returned {len(race_results)} drivers instead of 22.")

    weather = load_race_weather(2026, race_round, race_name)
    status = race_results["status"].astype(str)
    positions = pd.to_numeric(race_results["position"]).astype(int)
    race_frames.append(pd.DataFrame({
        "season": 2026,
        "round": race_round,
        "race_date": race_info["raceDate"],
        "grand_prix": race_name,
        "circuit_id": race_info["circuitId"],
        "circuit_name": race_info["circuitName"],
        "circuit_location": race_info["locality"],
        "country": race_info["country"],
        "driver_id": race_results["driverId"],
        "driver_code": race_results["driverCode"].fillna(""),
        "driver_name": race_results["givenName"].str.strip() + " " + race_results["familyName"].str.strip(),
        "constructor_id": race_results["constructorId"],
        "constructor": race_results["constructorName"],
        "weather_air_temp_c": weather["weather_air_temp_c"],
        "weather_track_temp_c": weather["weather_track_temp_c"],
        "weather_humidity_pct": weather["weather_humidity_pct"],
        "weather_wind_speed_mps": weather["weather_wind_speed_mps"],
        "weather_rainfall": weather["weather_rainfall"],
        "win": positions.eq(1).astype(int),
        "actual_finish_position": positions,
        "classified_position": race_results["positionText"],
        "points_scored": pd.to_numeric(race_results["points"]).astype(float),
        "status": status,
        "podium": positions.le(3).astype(int),
        "dnf": (~status.str.contains(r"Finished|\+\d+ Lap", case=False, regex=True)).astype(int),
        "weather_observations": weather["weather_observations"],
        "weather_source": weather["weather_source"],
    }))

if not race_frames:
    raise ValueError("No completed 2026 races were found.")
raw_2026 = pd.concat(race_frames, ignore_index=True)
print(f"Completed races returned: {raw_2026['round'].nunique()}")


def rebuild_historical_features(outcome_data):
    data = outcome_data.sort_values(
        ["season", "round", "actual_finish_position"]
    ).reset_index(drop=True).copy()
    driver_group = data.groupby("driver_id", sort=False)
    data["driver_prior_starts"] = driver_group.cumcount()

    rolling_features = {
        "driver_avg_finish_last5": ("actual_finish_position", "mean"),
        "driver_points_last5": ("points_scored", "sum"),
        "driver_wins_last5": ("win", "sum"),
        "driver_podiums_last5": ("podium", "sum"),
        "driver_dnf_rate_last5": ("dnf", "mean"),
    }
    for new_column, (source, calculation) in rolling_features.items():
        data[new_column] = driver_group[source].transform(
            lambda values: getattr(values.shift(1).rolling(5, min_periods=1), calculation)()
        )

    driver_season = data.groupby(["season", "driver_id"], sort=False)
    data["driver_season_points_before"] = (
        driver_season["points_scored"].cumsum() - data["points_scored"]
    )
    data["driver_season_wins_before"] = driver_season["win"].cumsum() - data["win"]
    data["driver_championship_rank_before"] = data.groupby(race_keys)[
        "driver_season_points_before"
    ].rank(method="min", ascending=False).astype(int)

    driver_circuit = data.groupby(["driver_id", "circuit_id"], sort=False)
    data["driver_circuit_starts_before"] = driver_circuit.cumcount()
    data["driver_avg_finish_at_circuit"] = driver_circuit[
        "actual_finish_position"
    ].transform(lambda values: values.shift(1).expanding().mean())
    data["driver_wins_at_circuit"] = driver_circuit["win"].cumsum() - data["win"]

    data["_wet_finish"] = data["actual_finish_position"] * data["weather_rainfall"]
    wet_group = data.groupby("driver_id", sort=False)
    data["driver_wet_starts_before"] = (
        wet_group["weather_rainfall"].cumsum() - data["weather_rainfall"]
    )
    wet_finish_before = wet_group["_wet_finish"].cumsum() - data["_wet_finish"]
    data["driver_avg_finish_in_wet_before"] = (
        wet_finish_before / data["driver_wet_starts_before"].replace(0, np.nan)
    )

    team_races = data.groupby(
        ["season", "round", "constructor_id"], as_index=False
    ).agg(
        race_points=("points_scored", "sum"),
        race_wins=("win", "sum"),
        race_dnfs=("dnf", "sum"),
    ).sort_values(["season", "round"])
    team_races["race_dnf_rate"] = team_races["race_dnfs"] / 2
    constructor_group = team_races.groupby("constructor_id", sort=False)
    constructor_rolling = {
        "constructor_points_last5": ("race_points", "sum"),
        "constructor_wins_last5": ("race_wins", "sum"),
        "constructor_dnf_rate_last5": ("race_dnf_rate", "mean"),
    }
    for new_column, (source, calculation) in constructor_rolling.items():
        team_races[new_column] = constructor_group[source].transform(
            lambda values: getattr(values.shift(1).rolling(5, min_periods=1), calculation)()
        )

    constructor_season = team_races.groupby(["season", "constructor_id"], sort=False)
    team_races["constructor_season_points_before"] = (
        constructor_season["race_points"].cumsum() - team_races["race_points"]
    )
    team_races["constructor_season_wins_before"] = (
        constructor_season["race_wins"].cumsum() - team_races["race_wins"]
    )
    team_races["constructor_championship_rank_before"] = team_races.groupby(race_keys)[
        "constructor_season_points_before"
    ].rank(method="dense", ascending=False).astype(int)

    constructor_columns = [
        "season", "round", "constructor_id", "constructor_points_last5",
        "constructor_wins_last5", "constructor_dnf_rate_last5",
        "constructor_season_points_before", "constructor_season_wins_before",
        "constructor_championship_rank_before",
    ]
    data = data.merge(
        team_races[constructor_columns],
        on=["season", "round", "constructor_id"],
        how="left",
        validate="many_to_one",
    )
    return data.drop(columns="_wet_finish")


historical_outcomes = cleaned_data[raw_2026.columns].copy()
combined_outcomes = pd.concat([historical_outcomes, raw_2026], ignore_index=True)
updated_data = rebuild_historical_features(combined_outcomes)[cleaned_data.columns]

engineered_features = [
    "driver_prior_starts", "driver_avg_finish_last5", "driver_points_last5",
    "driver_wins_last5", "driver_podiums_last5", "driver_dnf_rate_last5",
    "driver_season_points_before", "driver_season_wins_before",
    "driver_championship_rank_before", "driver_circuit_starts_before",
    "driver_avg_finish_at_circuit", "driver_wins_at_circuit",
    "driver_wet_starts_before", "driver_avg_finish_in_wet_before",
    "constructor_points_last5", "constructor_wins_last5",
    "constructor_dnf_rate_last5", "constructor_season_points_before",
    "constructor_season_wins_before", "constructor_championship_rank_before",
]
key_columns = ["season", "round", "driver_id"]
validation_rebuild = cleaned_data[key_columns + engineered_features].merge(
    updated_data.loc[updated_data["season"] <= 2025, key_columns + engineered_features],
    on=key_columns,
    suffixes=("_original", "_rebuilt"),
    validate="one_to_one",
)
validation_rows = []
for feature in engineered_features:
    original = pd.to_numeric(validation_rebuild[f"{feature}_original"], errors="coerce")
    rebuilt = pd.to_numeric(validation_rebuild[f"{feature}_rebuilt"], errors="coerce")
    mismatches = ~np.isclose(original, rebuilt, equal_nan=True, atol=1e-8)
    validation_rows.append({"feature": feature, "mismatched_rows": int(mismatches.sum())})
feature_validation = pd.DataFrame(validation_rows)

race_audit = raw_2026.groupby(
    ["round", "race_date", "grand_prix"], as_index=False
).agg(driver_rows=("driver_id", "size"), rainfall=("weather_rainfall", "first"))
baku_rows = raw_2026["grand_prix"].str.contains("Azerbaijan", case=False, na=False).sum()
winner_issues = raw_2026.groupby(race_keys)["win"].sum().ne(1).sum()
feature_mismatches = feature_validation["mismatched_rows"].sum()
expected_rows = raw_2026["round"].nunique() * 22
update_audit = pd.DataFrame({
    "check": [
        "Retrieval timestamp recorded", "Completed 2026 races added",
        "Expected 2026 driver rows", "Completed 2026 driver rows added",
        "Baku outcome rows included", "Races without exactly one winner",
        "Races without exactly 22 drivers", "Historical feature mismatches",
        "Maximum 2026 finishing position",
    ],
    "value": [
        retrieval_time.isoformat(), raw_2026["round"].nunique(), expected_rows,
        len(raw_2026), baku_rows, winner_issues,
        race_audit["driver_rows"].ne(22).sum(), feature_mismatches,
        raw_2026["actual_finish_position"].max(),
    ],
})

raw_2026.to_csv(TABLES_DIR / "part12_completed_2026_raw.csv", index=False)
updated_data.to_csv(TABLES_DIR / "part12_updated_dataset.csv", index=False)
race_audit.to_csv(TABLES_DIR / "part12_2026_race_audit.csv", index=False)
feature_validation.to_csv(TABLES_DIR / "part12_feature_rebuild_validation.csv", index=False)
update_audit.to_csv(TABLES_DIR / "part12_update_audit.csv", index=False)

print("\n2026 race audit\n" + "-" * 55)
print(race_audit.to_string(index=False))
print("\nUpdate validation\n" + "-" * 55)
print(update_audit.to_string(index=False))

if len(raw_2026) != expected_rows or race_audit["driver_rows"].ne(22).any():
    raise ValueError("Every completed 2026 race must contain exactly 22 drivers.")
if baku_rows or winner_issues or feature_mismatches:
    raise ValueError("Part 12 audit failed; review Baku, winner or feature validation checks.")
print("\nPart 12 completed successfully.")
print("Completed 2026 races were added with 22 drivers each.")
print("Baku outcome information remains absent.")
print("No qualifying, grid or practice fields were retained.")


# %% PART 13 — BAKU FORECAST DATASET

heading("PART 13 — BAKU FORECAST DATASET")

PROVISIONAL_ENTRANTS_APPROVED = True  # Approved by the project author.
freeze_time = pd.Timestamp.now(tz="Europe/Berlin")
baku_schedule = race_schedule[
    race_schedule["raceName"].str.contains("Azerbaijan", case=False, na=False)
]
if len(baku_schedule) != 1:
    raise ValueError("A unique 2026 Azerbaijan race was not found.")
baku_information = baku_schedule.iloc[0]
baku_round = int(baku_information["round"])
baku_date = pd.to_datetime(baku_information["raceDate"])

completed_2026 = updated_data[updated_data["season"] == 2026]
latest_round = int(completed_2026["round"].max())
entrants = completed_2026[completed_2026["round"] == latest_round][[
    "driver_id", "driver_code", "driver_name", "constructor_id", "constructor"
]].drop_duplicates("driver_id").sort_values(["constructor", "driver_name"]).reset_index(drop=True)
if len(entrants) != 22 or not entrants.groupby("constructor_id")["driver_id"].size().eq(2).all():
    raise ValueError("The provisional list must contain 22 drivers and two per constructor.")
if not PROVISIONAL_ENTRANTS_APPROVED:
    raise ValueError("The provisional entrant list has not been approved.")

driver_points_2026 = completed_2026.groupby("driver_id")["points_scored"].sum().reindex(
    entrants["driver_id"], fill_value=0
)
driver_ranks = driver_points_2026.rank(method="min", ascending=False).astype(int)
constructor_points_2026 = completed_2026.groupby("constructor_id")["points_scored"].sum().reindex(
    entrants["constructor_id"].unique(), fill_value=0
)
constructor_ranks = constructor_points_2026.rank(method="dense", ascending=False).astype(int)

feature_rows = []
for _, entrant in entrants.iterrows():
    driver_history = updated_data[
        updated_data["driver_id"] == entrant["driver_id"]
    ].sort_values(["season", "round"])
    recent_driver = driver_history.tail(5)
    circuit_history = driver_history[driver_history["circuit_id"] == "baku"]
    wet_history = driver_history[driver_history["weather_rainfall"] == 1]
    constructor_history = updated_data[
        updated_data["constructor_id"] == entrant["constructor_id"]
    ]
    constructor_races = constructor_history.groupby(
        ["season", "round"], as_index=False
    ).agg(
        race_points=("points_scored", "sum"),
        race_wins=("win", "sum"),
        race_dnfs=("dnf", "sum"),
    ).sort_values(["season", "round"])
    recent_constructor = constructor_races.tail(5)
    constructor_2026 = constructor_races[constructor_races["season"] == 2026]
    feature_rows.append({
        "season": 2026, "round": baku_round, "race_date": baku_date,
        "grand_prix": "Azerbaijan Grand Prix", "circuit_id": "baku",
        "circuit_name": "Baku City Circuit", "circuit_location": "Baku",
        "country": "Azerbaijan", "driver_id": entrant["driver_id"],
        "driver_code": entrant["driver_code"], "driver_name": entrant["driver_name"],
        "constructor_id": entrant["constructor_id"], "constructor": entrant["constructor"],
        "driver_prior_starts": len(driver_history),
        "driver_avg_finish_last5": recent_driver[TARGET].mean(),
        "driver_points_last5": recent_driver["points_scored"].sum(),
        "driver_wins_last5": recent_driver["win"].sum(),
        "driver_podiums_last5": recent_driver["podium"].sum(),
        "driver_dnf_rate_last5": recent_driver["dnf"].mean(),
        "driver_season_points_before": driver_points_2026[entrant["driver_id"]],
        "driver_season_wins_before": completed_2026.loc[
            completed_2026["driver_id"] == entrant["driver_id"], "win"
        ].sum(),
        "driver_championship_rank_before": driver_ranks[entrant["driver_id"]],
        "driver_circuit_starts_before": len(circuit_history),
        "driver_avg_finish_at_circuit": circuit_history[TARGET].mean(),
        "driver_wins_at_circuit": circuit_history["win"].sum(),
        "driver_wet_starts_before": len(wet_history),
        "driver_avg_finish_in_wet_before": wet_history[TARGET].mean(),
        "constructor_points_last5": recent_constructor["race_points"].sum(),
        "constructor_wins_last5": recent_constructor["race_wins"].sum(),
        "constructor_dnf_rate_last5": (recent_constructor["race_dnfs"] / 2).mean(),
        "constructor_season_points_before": constructor_2026["race_points"].sum(),
        "constructor_season_wins_before": constructor_2026["race_wins"].sum(),
        "constructor_championship_rank_before": constructor_ranks[entrant["constructor_id"]],
    })
baku_base = pd.DataFrame(feature_rows)

race_weather = updated_data.groupby(
    ["season", "round", "circuit_id"], as_index=False
).agg(
    weather_air_temp_c=("weather_air_temp_c", "first"),
    weather_track_temp_c=("weather_track_temp_c", "first"),
    weather_humidity_pct=("weather_humidity_pct", "first"),
    weather_wind_speed_mps=("weather_wind_speed_mps", "first"),
    weather_rainfall=("weather_rainfall", "first"),
)
weather_columns = [
    "weather_air_temp_c", "weather_track_temp_c", "weather_humidity_pct",
    "weather_wind_speed_mps",
]
baku_dry_history = race_weather[
    (race_weather["circuit_id"] == "baku") & (race_weather["weather_rainfall"] == 0)
]
wet_race_history = race_weather[race_weather["weather_rainfall"] == 1]
if baku_dry_history.empty or wet_race_history.empty:
    raise ValueError("Historical weather is insufficient for both scenarios.")

scenario_assumptions = pd.DataFrame([
    {
        "scenario": "Dry", **baku_dry_history[weather_columns].median().to_dict(),
        "weather_rainfall": 0,
        "assumption_basis": "Median weather from historical dry Baku races",
    },
    {
        "scenario": "Wet", **wet_race_history[weather_columns].median().to_dict(),
        "weather_rainfall": 1,
        "assumption_basis": "Median weather from historical wet races",
    },
])

scenario_frames = []
for _, scenario in scenario_assumptions.iterrows():
    scenario_data = baku_base.copy()
    scenario_data["scenario"] = scenario["scenario"]
    for column in weather_columns:
        scenario_data[column] = scenario[column]
    scenario_data["weather_rainfall"] = int(scenario["weather_rainfall"])
    scenario_data["weather_assumption"] = scenario["assumption_basis"]
    scenario_data["prediction_freeze_time"] = freeze_time.isoformat()
    scenario_data["entrant_status"] = (
        f"Provisional and author-approved: latest completed 2026 field, round {latest_round}"
    )
    scenario_frames.append(scenario_data)
baku_forecast_data = pd.concat(scenario_frames, ignore_index=True)

trained_features = list(final_model.feature_names_in_)
missing_features = sorted(set(trained_features) - set(APPROVED_FEATURES))
unexpected_features = sorted(set(APPROVED_FEATURES) - set(trained_features))
if missing_features or unexpected_features:
    raise ValueError(f"Missing features: {missing_features}; unexpected: {unexpected_features}")
approved_forecast_features = trained_features
leakage_found = [column for column in LEAKAGE_COLUMNS + [TARGET] if column in baku_forecast_data]
scenario_counts = baku_forecast_data.groupby("scenario")["driver_id"].nunique()
duplicate_scenarios = baku_forecast_data.duplicated(["scenario", "driver_id"]).sum()
if leakage_found or not scenario_counts.eq(22).all() or duplicate_scenarios:
    raise ValueError("Part 13 leakage, scenario-count or duplicate validation failed.")

baku_forecast_data.to_csv(TABLES_DIR / "part13_baku_forecast_dataset.csv", index=False)
baku_forecast_data[approved_forecast_features].to_csv(
    TABLES_DIR / "part13_baku_model_inputs.csv", index=False
)
scenario_assumptions.to_csv(TABLES_DIR / "part13_weather_scenarios.csv", index=False)

print("\nProvisional Baku entrants\n" + "-" * 55)
print(entrants[["driver_code", "driver_name", "constructor"]].to_string(index=False))
print("\nWeather scenario assumptions\n" + "-" * 55)
print(scenario_assumptions.round(2).to_string(index=False))
print("\nValidation checks\n" + "-" * 55)
print(f"Provisional entrants:        {len(entrants)}")
print(f"Constructors:                {entrants['constructor_id'].nunique()}")
print(f"Dry scenario rows:           {scenario_counts.get('Dry', 0)}")
print(f"Wet scenario rows:           {scenario_counts.get('Wet', 0)}")
print(f"Approved features:           {len(approved_forecast_features)}")
print("Feature order matches model: True")
print(f"Leakage columns found:       {leakage_found}")
print(f"Duplicate scenario rows:     {duplicate_scenarios}")
print(f"Prediction freeze time:      {freeze_time}")
print("\nPart 13 completed successfully.")
print("No Baku predictions have been generated.")


# %% PART 14 — FINAL BAKU PREDICTIONS

heading("PART 14 — FINAL BAKU PREDICTIONS")

# The model design is already locked. Refit it on all completed results for deployment.
deployment_model = clone(final_model)
thread_settings = {
    name: 1 for name in deployment_model.get_params()
    if name == "n_jobs" or name.endswith("__n_jobs")
}
deployment_model.set_params(**thread_settings)
plt.close("all")
deployment_model.fit(updated_data[approved_forecast_features], updated_data[TARGET])
joblib.dump(deployment_model, MODELS_DIR / "part14_final_forecast_model.joblib")

baku_forecast_data["predicted_score"] = deployment_model.predict(
    baku_forecast_data[approved_forecast_features]
)
baku_forecast_data = baku_forecast_data.sort_values(
    ["scenario", "predicted_score", "driver_name"]
).reset_index(drop=True)
baku_forecast_data["predicted_position"] = (
    baku_forecast_data.groupby("scenario").cumcount() + 1
)
baku_forecast_data["predicted_position_label"] = (
    "P" + baku_forecast_data["predicted_position"].astype(str)
)
scenario_type = pd.CategoricalDtype(["Dry", "Wet"], ordered=True)
baku_forecast_data["scenario"] = baku_forecast_data["scenario"].astype(scenario_type)
baku_forecast_data = baku_forecast_data.sort_values(
    ["scenario", "predicted_position"]
).reset_index(drop=True)

position_check = baku_forecast_data.groupby("scenario", observed=True)[
    "predicted_position"
].apply(lambda values: sorted(values) == list(range(1, 23)))
if not position_check.all() or baku_forecast_data.duplicated(["scenario", "driver_id"]).any():
    raise ValueError("Each scenario must contain one unique P1–P22 ranking.")

forecast_summary_rows = []
for scenario in ["Dry", "Wet"]:
    ranking = baku_forecast_data[baku_forecast_data["scenario"] == scenario]
    forecast_summary_rows.append({
        "scenario": scenario,
        "predicted_winner": ranking.iloc[0]["driver_name"],
        "predicted_podium": ", ".join(ranking.head(3)["driver_name"]),
        "predicted_top_ten": ", ".join(ranking.head(10)["driver_name"]),
    })
forecast_summary = pd.DataFrame(forecast_summary_rows)

position_comparison = baku_forecast_data.pivot(
    index=["driver_id", "driver_name", "constructor"],
    columns="scenario",
    values=["predicted_position", "predicted_score"],
).reset_index()
position_comparison.columns = [
    "_".join(str(value) for value in column if str(value)).strip("_")
    if isinstance(column, tuple) else column
    for column in position_comparison.columns
]
position_comparison = position_comparison.rename(columns={
    "predicted_position_Dry": "dry_position",
    "predicted_position_Wet": "wet_position",
    "predicted_score_Dry": "dry_score",
    "predicted_score_Wet": "wet_score",
})
position_comparison["positions_gained_in_wet"] = (
    position_comparison["dry_position"] - position_comparison["wet_position"]
)
position_comparison["score_change_wet_minus_dry"] = (
    position_comparison["wet_score"] - position_comparison["dry_score"]
)
position_comparison = position_comparison.sort_values(
    ["positions_gained_in_wet", "driver_name"], ascending=[False, True]
)

forecast_columns = [
    "scenario", "predicted_position", "predicted_position_label", "driver_code",
    "driver_name", "constructor", "predicted_score", "weather_air_temp_c",
    "weather_track_temp_c", "weather_humidity_pct", "weather_wind_speed_mps",
    "weather_rainfall", "weather_assumption", "prediction_freeze_time", "entrant_status",
]
baku_forecast_data[forecast_columns].to_csv(
    TABLES_DIR / "part14_baku_forecast.csv", index=False
)
forecast_summary.to_csv(TABLES_DIR / "part14_forecast_summary.csv", index=False)
position_comparison.to_csv(TABLES_DIR / "part14_weather_position_changes.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 12), sharex=True)
position_colours = {1: "#FFD700", 2: "#BFC1C2", 3: "#CD7F32"}
maximum_score = baku_forecast_data["predicted_score"].max() + 1
for axis, scenario in zip(axes, ["Dry", "Wet"]):
    ranking = baku_forecast_data[baku_forecast_data["scenario"] == scenario]
    colours = [
        position_colours.get(position, F1_RED if position <= 10 else F1_BLUE)
        for position in ranking["predicted_position"]
    ]
    labels = ranking["predicted_position_label"] + "  " + ranking["driver_name"]
    axis.barh(labels, ranking["predicted_score"], color=colours)
    axis.invert_yaxis()
    axis.set_xlim(0, maximum_score)
    axis.set_title(f"{scenario}-Weather Scenario", fontsize=17, weight="bold")
    axis.set_xlabel("Predicted finishing-position score — lower is better")
    axis.set_ylabel("Driver")
    axis.grid(axis="x", alpha=0.3)
fig.suptitle("Predicted 2026 Azerbaijan Grand Prix Rankings", fontsize=22, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_figure("part14_baku_dry_wet_rankings.png")

for scenario in ["Dry", "Wet"]:
    ranking = baku_forecast_data[baku_forecast_data["scenario"] == scenario][[
        "predicted_position_label", "driver_name", "constructor", "predicted_score"
    ]]
    print(f"\n{scenario} scenario ranking\n" + "-" * 55)
    print(ranking.round({"predicted_score": 3}).to_string(index=False))
print("\nForecast summary\n" + "-" * 55)
print(forecast_summary[["scenario", "predicted_winner", "predicted_podium"]].to_string(index=False))
print("\nLargest position changes in wet conditions\n" + "-" * 55)
print(position_comparison[[
    "driver_name", "dry_position", "wet_position", "positions_gained_in_wet"
]].head(10).to_string(index=False))
print("\nValidation\n" + "-" * 55)
print(f"Deployment training rows: {len(updated_data):,}")
print(f"Forecast rows:            {len(baku_forecast_data)}")
print(f"Dry positions valid:      {position_check['Dry']}")
print(f"Wet positions valid:      {position_check['Wet']}")
print("Baku target used:         False")
print("Qualifying/grid used:     False")
print("\nPart 14 completed successfully.")
print("These rankings are probabilistic analytical forecasts, not guaranteed race outcomes.")

# %% PART 15 — FINAL OUTPUT AUDIT

from pathlib import Path
import pandas as pd

print("\nPART 15 — FINAL OUTPUT AUDIT")
print("=" * 55)

PROJECT_DIR = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
MODELS_DIR = PROJECT_DIR / "outputs" / "models"

required_outputs = {
    "Cleaned dataset": [
        TABLES_DIR / "part2_cleaned_dataset.csv"
    ],
    "Cleaning summary": [
        TABLES_DIR / "part2_cleaning_summary.csv"
    ],
    "Data dictionary": [
        TABLES_DIR / "part3_data_dictionary.csv"
    ],
    "Descriptive statistics": [
        TABLES_DIR / "part4_descriptive_statistics.csv"
    ],
    "EDA figures": [
        FIGURES_DIR / "part4_figure1_finish_distribution.png",
        FIGURES_DIR / "part4_figure2_driver_recent_form.png",
        FIGURES_DIR / "part4_figure3_constructor_form.png",
        FIGURES_DIR / "part4_figure4_championship_rank.png",
        FIGURES_DIR / "part4_figure5_dnf_history.png",
        FIGURES_DIR / "part4_figure6_weather_relationship.png",
        FIGURES_DIR / "part4_figure7_correlation_heatmap.png",
        FIGURES_DIR / "part4_figure8_baku_history.png"
    ],
    "Regression models": [
        MODELS_DIR / "part6_ridge_model.joblib",
        MODELS_DIR / "part7_decision_tree_model.joblib",
        MODELS_DIR / "part8_random_forest_model.joblib"
    ],
    "Model results": [
        TABLES_DIR / "part6_baseline_ridge_results.csv",
        TABLES_DIR / "part7_decision_tree_results.csv",
        TABLES_DIR / "part8_random_forest_results.csv",
        TABLES_DIR / "part9_model_comparison.csv",
        TABLES_DIR / "part9_final_test_results.csv"
    ],
    "Regression diagnostics": [
        FIGURES_DIR / "part6_ridge_diagnostics.png",
        FIGURES_DIR / "part9_model_comparison.png"
    ],
    "Race-ranking evaluation": [
        TABLES_DIR / "part10_driver_rank_predictions.csv",
        TABLES_DIR / "part10_race_level_performance.csv",
        FIGURES_DIR / "part10_predicted_vs_actual_rank.png",
        FIGURES_DIR / "part10_position_error_distribution.png"
    ],
    "Feature importance": [
        TABLES_DIR / "part11_feature_importance.csv",
        TABLES_DIR / "part11_feature_group_importance.csv",
        FIGURES_DIR / "part11_feature_group_importance.png"
    ],
    "2026 update audit": [
        TABLES_DIR / "part12_updated_dataset.csv",
        TABLES_DIR / "part12_2026_race_audit.csv",
        TABLES_DIR / "part12_update_audit.csv",
        TABLES_DIR / "part12_feature_rebuild_validation.csv"
    ],
    "Baku forecast inputs": [
        TABLES_DIR / "part13_baku_forecast_dataset.csv",
        TABLES_DIR / "part13_weather_scenarios.csv"
    ],
    "Final Baku forecast": [
        TABLES_DIR / "part14_baku_forecast.csv",
        TABLES_DIR / "part14_forecast_summary.csv",
        TABLES_DIR / "part14_weather_position_changes.csv",
        FIGURES_DIR / "part14_baku_dry_wet_rankings.png",
        MODELS_DIR / "part14_final_forecast_model.joblib"
    ]
}

audit_rows = []

for output_group, paths in required_outputs.items():
    for path in paths:
        audit_rows.append({
            "output_group": output_group,
            "file_name": path.name,
            "folder": path.parent.name,
            "status": "Present" if path.exists() else "Missing",
            "file_size_kb":
                round(path.stat().st_size / 1024, 1)
                if path.exists() else 0
        })

output_audit = pd.DataFrame(audit_rows)

present_files = output_audit["status"].eq("Present").sum()
missing_files = output_audit["status"].eq("Missing").sum()


# Validate the final forecast itself.
forecast_path = TABLES_DIR / "part14_baku_forecast.csv"

forecast_checks = []

if forecast_path.exists():

    forecast = pd.read_csv(forecast_path)

    for scenario in ["Dry", "Wet"]:
        scenario_data = forecast[
            forecast["scenario"] == scenario
        ]

        positions_valid = (
            sorted(scenario_data["predicted_position"].tolist())
            == list(range(1, 23))
        )

        forecast_checks.append({
            "check": f"{scenario} scenario contains 22 drivers",
            "result": len(scenario_data) == 22
        })

        forecast_checks.append({
            "check": f"{scenario} scenario contains P1–P22",
            "result": positions_valid
        })

        forecast_checks.append({
            "check": f"{scenario} scenario has one predicted winner",
            "result":
                scenario_data["predicted_position"].eq(1).sum() == 1
        })

    forecast_checks.extend([
        {
            "check": "No duplicate driver-scenario rows",
            "result": not forecast.duplicated(
                ["scenario", "driver_name"]
            ).any()
        },
        {
            "check": "No Baku target information included",
            "result":
                "actual_finish_position" not in forecast.columns
        },
        {
            "check": "Forecast contains both weather scenarios",
            "result":
                set(forecast["scenario"]) == {"Dry", "Wet"}
        }
    ])

else:
    forecast_checks.append({
        "check": "Final forecast file exists",
        "result": False
    })

forecast_validation = pd.DataFrame(forecast_checks)


# Map saved evidence to the assignment marking criteria.
assignment_mapping = pd.DataFrame([
    {
        "assignment_criterion":
            "Introduction and predictive-analytics justification",
        "marks": 10,
        "main_evidence":
            "Research question, chronological design and race-day prediction scope",
        "code_evidence_status":
            "Ready",
        "report_action":
            "Write the introduction and justify regression and ranking."
    },
    {
        "assignment_criterion":
            "Dataset selection, cleaning, encoding and EDA",
        "marks": 25,
        "main_evidence":
            "Parts 1–5: cleaning summary, data dictionary, statistics and eight EDA figures",
        "code_evidence_status":
            "Complete",
        "report_action":
            "Select the most relevant tables and figures for the report."
    },
    {
        "assignment_criterion":
            "Machine-learning models and Python explanation",
        "marks": 25,
        "main_evidence":
            "Median baseline, Ridge, Decision Tree and Random Forest pipelines",
        "code_evidence_status":
            "Complete",
        "report_action":
            "Explain model suitability, preprocessing and limitations."
    },
    {
        "assignment_criterion":
            "Evaluation, visualisation, comparison and recommendation",
        "marks": 25,
        "main_evidence":
            "Parts 6–11: MAE, RMSE, R², Spearman and race-ranking evaluation",
        "code_evidence_status":
            "Complete",
        "report_action":
            "Recommend Random Forest using multiple validation metrics."
    },
    {
        "assignment_criterion":
            "Conclusion and implications",
        "marks": 10,
        "main_evidence":
            "2025 test results and dry/wet P1–P22 Baku forecasts",
        "code_evidence_status":
            "Ready",
        "report_action":
            "Write a balanced conclusion without claiming certainty."
    },
    {
        "assignment_criterion":
            "Harvard references",
        "marks": 5,
        "main_evidence":
            "Assignment brief, FastF1, Jolpica and academic ML sources",
        "code_evidence_status":
            "Report task",
        "report_action":
            "Add complete Harvard citations and matching in-text references."
    }
])


remaining_tasks = pd.DataFrame({
    "remaining_task": [
        "Write and edit the 3,000-word report",
        "Add Harvard in-text citations and reference list",
        "Declare AI assistance if required by the university policy",
        "Select final report figures and tables",
        "Rerun Parts 12–14 if more races are completed before submission",
        "Replace the provisional entrant list if an official Baku list appears"
    ]
})


# Check that the script contains Parts 0–15.
script_path = Path(__file__).resolve()
script_text = script_path.read_text(encoding="utf-8")
section_count = script_text.count("# %% PART")

reproducibility_check = pd.DataFrame({
    "check": [
        "Python script exists",
        "Script contains Parts 0–15",
        "Random state fixed at 42",
        "Output folders use pathlib",
        "Final forecast validation passed"
    ],
    "result": [
        script_path.exists(),
        section_count >= 16,
        "RANDOM_STATE = 42" in script_text,
        "Path(" in script_text,
        forecast_validation["result"].all()
    ]
})


output_audit.to_csv(
    TABLES_DIR / "part15_output_audit.csv",
    index=False
)

forecast_validation.to_csv(
    TABLES_DIR / "part15_forecast_validation.csv",
    index=False
)

assignment_mapping.to_csv(
    TABLES_DIR / "part15_assignment_mapping.csv",
    index=False
)

remaining_tasks.to_csv(
    TABLES_DIR / "part15_remaining_tasks.csv",
    index=False
)

reproducibility_check.to_csv(
    TABLES_DIR / "part15_reproducibility_check.csv",
    index=False
)


print("\nOutput inventory")
print("-" * 55)
print(f"Required files present: {present_files}")
print(f"Required files missing: {missing_files}")

if missing_files:
    print("\nMissing files")
    print("-" * 55)
    print(
        output_audit.loc[
            output_audit["status"] == "Missing",
            ["output_group", "file_name"]
        ].to_string(index=False)
    )

print("\nFinal forecast validation")
print("-" * 55)
print(forecast_validation.to_string(index=False))

print("\nAssignment coverage")
print("-" * 55)
print(
    assignment_mapping[
        [
            "assignment_criterion",
            "marks",
            "code_evidence_status"
        ]
    ].to_string(index=False)
)

print("\nReproducibility checks")
print("-" * 55)
print(reproducibility_check.to_string(index=False))

print("\nRemaining report tasks")
print("-" * 55)
print(remaining_tasks.to_string(index=False))

if missing_files == 0 and forecast_validation["result"].all():
    print("\nPart 15 completed successfully.")
    print("All required analytical outputs are present.")
else:
    print("\nPart 15 identified items that still require attention.")
