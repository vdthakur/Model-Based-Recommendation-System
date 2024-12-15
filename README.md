# Model-Based Recommendation System with XGBoost

## Overview

This project predicts Yelp user ratings for businesses using a model-based recommendation system. By integrating data from multiple Yelp datasets and applying advanced feature engineering techniques, the system accurately models the interaction between users and businesses. The final model is powered by **XGBoost** with optimized hyperparameters, achieving a **Root Mean Square Error (RMSE)** of approximately **0.9743** on the test set.

## Key Highlights

- **Advanced Feature Engineering**: Combines user behaviors, business characteristics, spatial transformations, temporal data, and user-business interactions into meaningful features.
- **Multi-Dataset Integration**: Leverages Yelp datasets, including user, business, check-in, tip, and photo data.
- **State-of-the-Art Model**: Utilizes **XGBoost**, a gradient boosting framework, for efficient and accurate predictions.
- **Hyperparameter Optimization**: Incorporates **Optuna** to fine-tune XGBoost hyperparameters for optimal performance.

---

## Features and Data

### 1. **Features Used**
- **User Features**:
  - Total number of reviews and average rating.
  - Aggregated counts of compliments (e.g., `useful`, `funny`, `cool`, `fans`).
  - Time since the user joined Yelp, representing engagement duration.
  
- **Business Features**:
  - Total number of reviews, average rating, and whether the business is open.
  - Common business attributes (e.g., accepts credit cards, kid-friendly, delivery, reservations).
  - Spatial features: Latitude and longitude converted into sine and cosine values to represent location.

- **Interaction Features**:
  - Average likes on tips by user and business.
  - Check-in frequencies for businesses.
  - Photo counts per business.

- **Categorical Encodings**:
  - Encoded attributes like **NoiseLevel** (e.g., `quiet`, `average`, `loud`) into numerical values. After experimentation, only **NoiseLevel** improved the RMSE and was included in the final model.

---

### 2. **Datasets Used**
- **`yelp_train.csv`**: Training data linking users to businesses and their ratings.
- **`yelp_val.csv`**: Test data used for prediction and evaluation.
- **`user.json`**: Contains user information, including review counts, average stars, and compliments.
- **`business.json`**: Details about businesses, such as location, attributes, and total reviews.
- **`checkin.json`**: Check-in frequency data for businesses.
- **`tip.json`**: Tips left by users, with corresponding like counts.
- **`photo.json`**: Number of photos associated with each business.

---

## Model Training & Optimization

- **Algorithm**: [XGBoost](https://xgboost.ai/), a gradient boosting framework, was used for its ability to handle sparse data and its scalability.
- **Hyperparameter Tuning**:
  - Used **Optuna**, an advanced optimization framework, to tune parameters such as `learning_rate`, `max_depth`, `subsample`, `gamma`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`.

---

## Results & Performance

### Model Evaluation
- **RMSE**: ~0.9743 on the test set.
- **Error Distribution**:
  - Predictions within 1 star of the true rating: **102,587**
  - Predictions within 2 stars of the true rating: **32,515**
  - Predictions within 3 stars of the true rating: **6,148**
  - Predictions within 4 stars of the true rating: **794**
  - Predictions outside 4 stars: **0**

### Execution Time
- **Local Environment**: ~42 seconds.
- **Cloud Environment**: ~523 seconds.

---

### Parameters
The script accepts the following parameters:

- `<folder_path>`: Path to the directory containing all input files (e.g., `yelp_train.csv`, `user.json`, etc.).
- `<testing_set>`: Path to the test CSV file (e.g., `yelp_test.csv`).
- `<output>`: Path to the output file where predictions will be saved.

### Example Command
To run the recommendation system, use the following command:

```bash
python recommendation_system.py data yelp_test.csv predictions.csv
```
---
