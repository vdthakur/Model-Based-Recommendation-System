# Yelp Rating Prediction Using Model-Based Recommendation with XGBoost

## Project Overview

This project aims to predict Yelp user ratings for businesses using a model-based recommendation system. By integrating multiple Yelp datasets and employing feature engineering, the system leverages user behavior, business characteristics, spatial details, and temporal factors to produce accurate rating predictions. The final model, trained using XGBoost with hyperparameter optimization, achieves a Root Mean Square Error (RMSE) of approximately **0.97599** on the validation set.

## Key Features

- **Comprehensive Feature Engineering:**  
  - **User features:** Aggregated metrics such as total number of reviews, average ratings, and combined counts of all user compliments and reactions (useful/funny/cool/fans).
  - **Business attributes:** Captured common attributes (payment options, price range, children-friendly, takeout availability, delivery options, outdoor seating, etc.) and encoded True/False values as binary features.
  - **Spatial features:** Transformed business latitude and longitude coordinates into sine and cosine values for a more nuanced spatial representation.
  - **Temporal features:** Calculated time since a user’s first Yelp activity to represent user engagement duration.
  - **Interaction features:** Incorporated average tip likes per user and business, check-in frequency, and counts of photos associated with businesses.
  - **Categorical encodings:** Encoded certain categorical attributes (e.g., NoiseLevel) into numeric values. After experimentation, only NoiseLevel improved the model’s performance and was retained.

- **Data Sources:**
  - **`yelp_train.csv` & `yelp_val.csv` (or `yelp_test.csv`):** Interaction data linking users to businesses with corresponding ratings.
  - **`user.json`:** Information about users, including average stars, review counts, and compliment totals.
  - **`business.json`:** Business details such as total reviews, average rating, location coordinates, and attributes.
  - **`checkin.json`:** Business check-in frequencies.
  - **`tip.json`:** Tips data used to derive average likes per user and business.
  - **`photo.json`:** Photo counts per business.

- **Model Tuning & Evaluation:**  
  Used Optuna and XGBoost’s parameter tuning capabilities to optimize hyperparameters. The final model yields a RMSE of ~0.97599.

## Results & Performance

**RMSE:** ~0.97599 on the validation set.

**Error Distribution:**
- Predictions within 1 star of the true rating: 102,587
- Predictions within 2 stars of the true rating: 32,515
- Predictions within 3 stars of the true rating: 6,148
- Predictions within 4 stars of the true rating: 794
- Predictions outside of 4 stars: 0

**Execution Time:**
- Local environment: ~42 seconds
- Vocareum environment: ~523 seconds

## Repository Structure
