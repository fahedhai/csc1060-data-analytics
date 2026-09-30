"""
Student Performance Prediction (CSC1060 - Problem Solving 2)

Develops and compares three regression models (Linear Regression,
Decision Tree, Random Forest) to predict students' final grades from
demographic, attendance, study behaviour, and prior academic performance
data.

Dataset: Student Performance dataset (school, sex, age, address type,
weekly study time, past failures, absences, period 1/2 grades, final grade)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load and inspect data
# ---------------------------------------------------------------
df = pd.read_csv("student_performance.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 8 duplicate rows were found in the original dataset -- removed to avoid
# repeated observations skewing the models
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# ---------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Final Grade"], bins=20, kde=True)
plt.title("Distribution of Final Grade")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("figures/final_grade_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------
# One-hot encode categorical variables (School, Student Sex, Home Address
# Type), dropping the first category of each to avoid redundant columns
df_encoded = pd.get_dummies(
    df, columns=["School", "Student Sex", "Home Address Type"], drop_first=True
)

X = df_encoded.drop("Final Grade", axis=1)
y = df_encoded["Final Grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train and evaluate three models
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Results")
    print("-" * (len(name) + 8))
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.2f}")
    return mae, rmse, r2


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
mae_lin, rmse_lin, r2_lin = evaluate("Linear Regression", y_test, y_pred_linear)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mae_tree, rmse_tree, r2_tree = evaluate("Decision Tree", y_test, y_pred_tree)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
mae_forest, rmse_forest, r2_forest = evaluate("Random Forest", y_test, y_pred_forest)

# ---------------------------------------------------------------
# 5. Compare models
# ---------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [mae_lin, mae_tree, mae_forest],
    "RMSE": [rmse_lin, rmse_tree, rmse_forest],
    "R2 Score": [r2_lin, r2_tree, r2_forest],
}).sort_values(by="R2 Score", ascending=False)

print("\nModel Comparison:")
print(results)

# ---------------------------------------------------------------
# 6. Feature importance (Random Forest)
# ---------------------------------------------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest_model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="Importance", y="Feature")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Actual vs predicted + residuals (best model: Random Forest)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_forest, alpha=0.7)
plt.plot([0, 20], [0, 20], color="red", linestyle="--")
plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", dpi=150)
plt.close()

residuals = y_test - y_pred_forest
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_forest, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Final Grade")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("figures/residual_plot.png", dpi=150)
plt.close()

print("\nAll charts saved to figures/")
