# 📊 Predicting Student Final Grades

**Course:** CSC1060 — Data Analytics Fundamentals (Problem Solving 2)  
**Author:** Fahed Haidar — IT/Data Science Student

## Project Overview

A predictive data analysis project investigating which factors are associated with students' final academic grades.

The project uses student performance data to prepare and analyse demographic, attendance, study habit and prior academic performance variables. Three regression models were developed and compared to evaluate their ability to predict final grades.

The project demonstrates a complete data analytics and machine learning workflow, including:

- Data cleaning and preparation
- Exploratory data analysis
- Feature encoding and selection
- Feature scaling
- Regression modelling
- Model comparison
- Performance evaluation
- Interpretation of results


## Dataset

The project uses a student performance dataset containing **649 records and 10 variables** covering:

- School
- Sex
- Age
- Home address type
- Weekly study time
- Past class failures
- School absences
- First period grade
- Second period grade
- Final grade

During data preparation, **8 duplicate rows were removed**, leaving **641 unique records**. No missing values were identified.

## Approach

The analysis followed a structured machine learning workflow:

1. **Data preparation** — checked the dataset for duplicates and missing values.
2. **Feature preparation** — categorical variables were converted using one-hot encoding.
3. **Train/test split** — data was divided into 80% training and 20% testing sets.
4. **Feature scaling** — numerical features were scaled using `StandardScaler`.
5. **Model development** — three regression models were trained:
   - Linear Regression
   - Decision Tree
   - Random Forest
6. **Model evaluation** — models were compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

## Results

Three regression models were trained and evaluated using MAE, RMSE, and R² Score.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| **Linear Regression** | **0.76** | **1.01** | **0.88** |
| Random Forest | 0.82 | 1.21 | 0.82 |
| Decision Tree | 0.94 | 1.56 | 0.71 |

### Best Performing Model

**Linear Regression achieved the strongest overall performance**, with an R² score of **0.88**, an MAE of **0.76**, and an RMSE of **1.01**.

The strong performance is likely related to the near-linear relationship between final grades and prior academic performance, particularly first and second period grades. On this dataset of 641 records, the linear model captured most of the predictive signal without the additional variance introduced by the tree-based models.

## Key Findings

- **Prior academic performance was the strongest predictor of final grade.** First and second period grades had substantially stronger relationships with final grades than the demographic and behavioural variables examined.
- **Past class failures were negatively associated with final grades.** Students with a history of previous failures tended to achieve lower final grades.
- **Weekly study time showed only a weak positive relationship** with final grade.
- **Age and school absences showed relatively weak relationships** with final grade, suggesting that the variables included in this dataset do not fully explain differences in academic performance.
- The results demonstrate that **feature selection and model comparison are important** when developing predictive models, as the three regression approaches produced noticeably different performance.


## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python student_performance_analysis.py
```
Requires `student_performance.csv` in the same folder.

## Files
- `student_performance_analysis.py` — full analysis script
- `report.pdf` — full written report with methodology, findings, and evaluation

## Project Structure

```text
assessment-2-student-performance/
├── student_performance_analysis.py   # Data analysis and machine learning workflow
├── student_performance.csv           # Student performance dataset
├── report.pdf                         # Full analysis report
└── README.md                          # Project documentation
