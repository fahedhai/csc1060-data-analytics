# Assessment 2: Predicting Student Final Grades

**Course:** CSC1060 - Data Analytics Fundamentals (Problem Solving 2)

## Overview
Builds and compares three regression models to predict students' final
grades from demographic data, attendance, study habits, and prior academic
performance (first/second period grades).

## Dataset
Student performance dataset — 649 records, 10 variables (school, sex, age,
home address type, weekly study time, past class failures, school absences,
period 1/2 grades, final grade). 8 duplicate rows removed, leaving 641
unique records. No missing values.

## Approach
- One-hot encoded categorical variables (School, Student Sex, Home Address
  Type)
- 80/20 train/test split, features scaled with StandardScaler
- Compared three models: Linear Regression, Decision Tree, Random Forest

## Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Linear Regression** | **0.76** | **1.01** | **0.88** |
| Random Forest | 0.82 | 1.21 | 0.82 |
| Decision Tree | 0.94 | 1.56 | 0.71 |

Linear Regression performed best — likely because final grade has a very
strong, near-linear relationship with prior grades (first/second period),
so a simple linear model captures most of the signal without the added
variance that tree-based models can introduce on a relatively small
dataset (641 rows).

## Key Findings
- **Second Period Grade and First Period Grade are by far the strongest
  predictors** of final grade — prior academic performance dominates over
  demographic or behavioral factors.
- **Past class failures showed a moderate negative correlation** with final
  grade — students with a history of failures tend to score lower.
- Weekly study time had only a weak positive relationship with final grade,
  and age/absences showed weak correlations — somewhat counterintuitive,
  suggesting other unmeasured factors (motivation, support at home, etc.)
  may matter more than these simpler behavioral proxies.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python student_performance_analysis.py
```
Requires `student_performance.csv` in the same folder.

## Files
- `student_performance_analysis.py` — full analysis script
- `report.pdf` — full written report with methodology, findings, and evaluation
