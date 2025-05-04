# Kaggle House Price Prediction

Welcome to the House Price Prediction project! This repository showcases a complete data science pipeline for predicting house prices using the popular Kaggle dataset. The project is designed to demonstrate best practices in data preprocessing, feature engineering, model training, and evaluation, making it a valuable reference for MSc Data Science students and practitioners.

## Project Overview

This project aims to accurately predict house sale prices using various features from the Ames Housing dataset. The workflow includes:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model selection and training (CatBoost)
- Generating predictions for Kaggle submission

The approach is modular, reproducible, and suitable for both academic and practical applications.

---

## Dataset

- **Source:** [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Files:**
  - `train.csv`: Training data with features and target variable
  - `test.csv`: Test data for prediction
  - `data_description.txt`: Detailed feature descriptions
  - `sample_submission.csv`: Sample submission format

---

## Project Structure

```
├── house-price-prediction-project-1.ipynb   # Main analysis notebook
├── kaggle_preprocessing.py                  # Data preprocessing scripts
├── kaggle_modeling.py                       # Model training and prediction scripts
├── result.csv                               # Model predictions for submission
├── test.py                                  # Test script
├── train.csv, test.csv                      # Dataset files
├── data_description.txt                     # Feature descriptions
└── sample_submission.csv                    # Kaggle submission template
```

---

## Requirements

- Python 3.7+
- Jupyter Notebook
- pandas, numpy, scikit-learn
- catboost

Install dependencies with:
```bash
pip install -r requirements.txt
```
*(Create a `requirements.txt` as needed)*

---

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Joon-hub/Kaggle_House_Price_Prediction.git
   cd Kaggle_House_Price_Prediction
   ```

2. **Run the notebook:**
   Open `house-price-prediction-project-1.ipynb` in Jupyter Notebook and follow the workflow.

3. **Command-line usage:**
   Use `kaggle_preprocessing.py` and `kaggle_modeling.py` to preprocess data and train models.

4. **Generate predictions:**
   The final predictions are saved in `result.csv` for submission to Kaggle.

---

## Results

- The CatBoost model achieved competitive performance on the Kaggle leaderboard.
