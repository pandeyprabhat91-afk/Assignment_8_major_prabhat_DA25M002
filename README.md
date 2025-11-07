# Assignment 8: Ensemble Learning for Bike Share Data

**Course:** DA5401 - Data Analytics Lab  
**Name:** Maj Prabhat Pandey  
**Roll Number:** DA25M002  
**Program:** M.Tech (AI & DS), IIT Madras  

This project implements multiple ensemble learning algorithms for predicting daily bike rental counts using the UCI Bike Sharing dataset. The notebook demonstrates preprocessing, baseline modeling, ensemble methods, and model evaluation.

---

## Project Overview

The goal of this assignment is to compare ensemble learning techniques for regression tasks and analyze their ability to reduce bias and variance compared to a baseline model.

Models implemented:
- Linear Regression (Baseline)
- Bagging Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Stacking Regressor (meta-learner: Ridge Regression)

---

## Data Description

Dataset: **Bike Sharing Dataset (day.csv)**  
Source: UCI Machine Learning Repository

Key features:
- Season, year, month, weekday, weather situation
- Temperature, humidity, windspeed
- Target variable: `cnt` (total bike rentals per day)

Total records: 731  
Missing values: None  
All features are numeric after preprocessing.

---

## Methodology

### 1. Data Preprocessing
- Removed non-predictive and redundant columns: `instant`, `dteday`, `casual`, `registered`
- Scaled all features using StandardScaler
- 80–20 train-test split for evaluation

### 2. Baseline Model
- Linear Regression established baseline performance
  - Test RMSE: 831.29  
  - Test R2: 0.8277

### 3. Ensemble Models
| Model | Description | Test R2 | Test RMSE |
|--------|--------------|----------|------------|
| Bagging | Aggregates multiple Decision Trees | 0.8870 | 673.24 |
| Random Forest | Bagging with feature randomness | 0.8840 | 682.12 |
| AdaBoost | Sequentially adjusts sample weights | 0.8532 | 767.19 |
| Gradient Boosting | Gradient-based sequential boosting | 0.8943 | 651.03 |
| Stacking | Combines RF, GB, AdaBoost via Ridge meta-learner | **0.8989** | **636.71** |

---

## Model Evaluation

**Metrics used:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

**Best Performing Model:** Stacking Ensemble  
- Test R2 Score: 0.8989  
- Test RMSE: 636.71  
- Demonstrates ~9% improvement over Gradient Boosting and ~27% over Linear Regression.

**Feature Importance (Random Forest):**
1. Temperature (35.2%)
2. Year (27.9%)
3. Apparent Temperature (15.4%)
4. Humidity (5.9%)
5. Season (5.3%)

---

## Insights

1. Ensemble methods significantly outperform baseline regression.
2. Bagging and Random Forest reduce variance effectively.
3. Boosting techniques improve bias correction but risk overfitting.
4. Stacking provides optimal generalization by combining model strengths.
5. Temperature and year are key predictors of rental activity.

---

## Results Summary

| Model | Train R2 | Test R2 | RMSE | MAE | Overfitting |
|--------|-----------|---------|------|-----|--------------|
| Linear Regression | 0.791 | 0.828 | 831 | 617 | Good |
| Bagging | 0.982 | 0.887 | 673 | 425 | Moderate |
| Random Forest | 0.982 | 0.884 | 682 | 428 | Moderate |
| AdaBoost | 0.937 | 0.853 | 767 | 534 | Moderate |
| Gradient Boosting | 0.994 | 0.894 | 651 | 446 | Moderate |
| Stacking | 0.994 | **0.899** | **637** | **426** | Moderate |

---

## Tools and Libraries

- Python 3.10  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn (ensemble, metrics, preprocessing)

---
