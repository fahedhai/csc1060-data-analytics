"""
Arthritis Prevalence Analysis (CSC1060 - Problem Solving 1)

Analyzes arthritis prevalence among older adults using the CDC's
Alzheimer's Disease and Healthy Aging Data, exploring patterns by
location, sex, race/ethnicity, and year, then fits a simple linear
regression to see whether prevalence can be predicted from year alone.

Dataset: CDC Alzheimer's Disease and Healthy Aging Data
(https://data.cdc.gov/Healthy-Aging/Alzheimer-s-Disease-and-Healthy-Aging-Data/hfr9-rurv)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------
# 1. Load and filter data
# ---------------------------------------------------------------
df = pd.read_csv("Alzheimers_Disease_and_Healthy_Aging_Data.csv")

print(df.head())
print(df.info())
print(df["Topic"].unique())

arthritis_df = df[df["Topic"] == "Arthritis among older adults"]
print(f"\nArthritis records: {arthritis_df.shape}")

# ---------------------------------------------------------------
# 2. Select relevant columns and clean
# ---------------------------------------------------------------
arthritis = arthritis_df[[
    "LocationDesc",
    "YearStart",
    "Data_Value",
    "StratificationCategory1",
    "Stratification1",
    "StratificationCategory2",
    "Stratification2",
]]

print(f"\nMissing values:\n{arthritis.isnull().sum()}")

# Data_Value is the key metric (% of older adults with arthritis) -- drop
# rows missing it so all downstream stats/models use complete observations
arthritis = arthritis.dropna(subset=["Data_Value"])
print(f"\nSummary stats:\n{arthritis['Data_Value'].describe()}")

# ---------------------------------------------------------------
# 3. Arthritis prevalence by location
# ---------------------------------------------------------------
state_avg = arthritis.groupby("LocationDesc")["Data_Value"].mean().sort_values(ascending=False)
print(f"\nTop 10 states by arthritis prevalence:\n{state_avg.head(10)}")

plt.figure(figsize=(12, 6))
state_avg.head(10).plot(kind="bar")
plt.title("Top 10 States with Highest Arthritis Prevalence")
plt.xlabel("State")
plt.ylabel("Average Percentage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/arthritis_by_state.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Arthritis prevalence by sex
# ---------------------------------------------------------------
sex = arthritis[arthritis["StratificationCategory2"] == "Sex"]
sex_avg = sex.groupby("Stratification2")["Data_Value"].mean()
print(f"\nAverage prevalence by sex:\n{sex_avg}")

plt.figure(figsize=(6, 4))
sex_avg.plot(kind="bar")
plt.title("Average Arthritis Prevalence by Sex")
plt.xlabel("Sex")
plt.ylabel("Average Percentage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/arthritis_by_sex.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Arthritis prevalence by race/ethnicity
# ---------------------------------------------------------------
race = arthritis[arthritis["StratificationCategory2"] == "Race/Ethnicity"]
race_avg = race.groupby("Stratification2")["Data_Value"].mean().sort_values()
print(f"\nAverage prevalence by race/ethnicity:\n{race_avg}")

plt.figure(figsize=(10, 6))
race_avg.plot(kind="barh")
plt.title("Average Arthritis Prevalence by Race/Ethnicity")
plt.xlabel("Average Percentage (%)")
plt.ylabel("Race/Ethnicity")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/arthritis_by_race.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Arthritis prevalence over time
# ---------------------------------------------------------------
year_avg = arthritis.groupby("YearStart")["Data_Value"].mean()
print(f"\nYearly averages:\n{year_avg}")

plt.figure(figsize=(8, 5))
year_avg.plot(kind="line", marker="o")
plt.title("Average Arthritis Prevalence Over Time")
plt.xlabel("Year")
plt.ylabel("Average Percentage (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/arthritis_over_time.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Simple predictive model: can Year predict prevalence?
# ---------------------------------------------------------------
X = arthritis[["YearStart"]]
y = arthritis["Data_Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nLinear Regression (Year -> Arthritis Prevalence)")
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Arthritis Percentage")
plt.ylabel("Predicted Arthritis Percentage")
plt.title("Actual vs Predicted Arthritis Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", dpi=150)
plt.close()

print("\nAll charts saved to figures/")
