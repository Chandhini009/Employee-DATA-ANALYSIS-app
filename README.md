# Dyashin HR Insights (Final)- Employees Data Analysis

Dyashin HR Insights is an interactive Streamlit application for exploring HR datasets, analyzing workforce trends, and predicting employee attrition using a RandomForest classifier enhanced with SMOTE balancing.

---

Streamlit APP link: https://employee-data-analysis.streamlit.app/
---

## Features

- **Dashboard**
  - Gender distribution
  - Department-wise employee count
  - Tenure visualization

- **Salary Analysis**
  - Average salary by department and location
  - Salary distribution by designation and department

- **Attrition Prediction**
  - Preprocesses data (numeric scaling & categorical encoding)
  - Balances classes using SMOTE
  - Trains a RandomForest classifier
  - Shows model accuracy, confusion matrix, and feature importance
  - Predict attrition for new employees interactively

---

## Requirements

- Python 3.9+
- Dependencies (see `requirements.txt`)

---

## Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd <repo_folder>

#streamlit app
 
--- streamlit run emp.py
