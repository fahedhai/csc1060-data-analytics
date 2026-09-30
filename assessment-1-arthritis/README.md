# Assessment 1: Arthritis Prevalence Among Older Adults

**Course:** CSC1060 - Data Analytics Fundamentals (Problem Solving 1)

## Overview
Analyzes arthritis prevalence among older adults across the US using the
CDC's Alzheimer's Disease and Healthy Aging Data, exploring how prevalence
varies by location, sex, race/ethnicity, and year, then fits a linear
regression model to test whether year alone predicts prevalence.

## Dataset
CDC Alzheimer's Disease and Healthy Aging Data, filtered to the "Arthritis
among older adults" topic (6,053 records after filtering).

## Key Findings
- **West Virginia had the highest arthritis prevalence** (~57.3%), followed
  by Alabama (~53.3%) and Kentucky (~50.7%) — a clear regional pattern in
  the data (concentrated in the South/Appalachia).
- **Women had notably higher prevalence than men** (47.6% vs 38.2%),
  consistent with existing research on osteoarthritis risk factors
  (hormonal changes, longer life expectancy).
- **Native American/Alaskan Native populations had the highest prevalence
  by race/ethnicity**, followed by Black non-Hispanic and White non-Hispanic
  groups; Asian/Pacific Islander populations had the lowest.
- Prevalence over time was relatively stable (~41-44%) with a noticeable
  dip in 2020 and a rise in 2022.

## How to Run
```bash
pip install pandas matplotlib scikit-learn
python arthritis_analysis.py
```
Requires the CDC dataset CSV (`Alzheimers_Disease_and_Healthy_Aging_Data.csv`)
in the same folder — available from the
[CDC's public data portal](https://data.cdc.gov/Healthy-Aging/Alzheimer-s-Disease-and-Healthy-Aging-Data/hfr9-rurv).

## Files
- `arthritis_analysis.py` — full analysis script
- `report.pdf` — full written report with methodology, findings, and references
