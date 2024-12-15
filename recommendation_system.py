# This model based recommendation system uses Yelp data to predict user ratings for businesses.
# It integrates multiple datasets (user, business, check-in, photo, tip) and employs 
# advanced preprocessing and feature engineering techniques. The final model is an 
# XGBoost-based regression model trained to predict ratings.


import time
import sys
import csv
import math
from pyspark import SparkContext
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
import re

# Initialize Spark context for distributed processing.
recommendation_system = SparkContext('local[*]', 'recommendation_system')

# Read command line arguments for input and output file paths.
folder_path = sys.argv[1] 
testing_set = sys.argv[2] 
output = sys.argv[3] 

# Define file paths for all the input JSON and CSV files needed.
training_set = folder_path + "/yelp_train.csv"
user = folder_path + "/user.json"
business = folder_path + "/business.json"
checkin = folder_path + "/checkin.json"
tip = folder_path + "/tip.json"
photo = folder_path + "/photo.json"


# =============================
# Load and process training data
# =============================

train_rdd = recommendation_system.textFile(training_set)
first_line_train = train_rdd.first()
# Filter out the header row and parse the CSV rows into (user_id, business_id, rating).
train_rdd = train_rdd.filter(lambda line: line != first_line_train).map(lambda row: row.strip().split(','))
train_rdd = train_rdd.map(lambda row: (row[0], row[1], float(row[2])))

# ============================
# Load and process testing data
# ============================

test_rdd = recommendation_system.textFile(testing_set)
first_line_test = test_rdd.first()
# Filter out header row and parse the test CSV into (user_id, business_id).
test_rdd = test_rdd.filter(lambda line: line != first_line_test).map(lambda row: row.strip().split(',')).map(lambda row: (row[0], row[1]))

# ============================
# Load and process user dataset
# ============================

user_rdd_base = recommendation_system.textFile(user)

# Extract user-level features: review_count, average_stars, and yelping_since.
user_rdd = user_rdd_base.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (x.get('review_count', 0), x.get('average_stars', None), x.get('yelping_since', None)))) \
    .collectAsMap() 

# Sum up user interactions (useful, funny, cool, fans) for each user to get an overall user engagement metric.
user_rdd_features = user_rdd_base.map(lambda x: json.loads(x)).map(lambda user: (user['user_id'], user['useful'] + user['funny'] + user['cool'] + user['fans'])) \
    .collectAsMap()

# Sum up all user compliments into a single metric.
user_rdd_compliment = user_rdd_base.map(lambda x: json.loads(x)) \
    .map(lambda user: (
        user['user_id'],
        sum([user['compliment_hot'], user['compliment_more'], user['compliment_profile'],
             user['compliment_cute'], user['compliment_list'], user['compliment_note'],
             user['compliment_plain'], user['compliment_cool'], user['compliment_funny'],
             user['compliment_writer'], user['compliment_photos']]))).collectAsMap()

# ===============================
# Load and process business dataset
# ===============================

business_rdd_base = recommendation_system.textFile(business)

# Extract simple business metrics: review_count and average_stars.
business_rdd = business_rdd_base.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x.get('review_count', 0), x.get('stars', None)))).collectAsMap()

# Extract business "is_open" status.
business_is_open = business_rdd_base.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['is_open'])).collectAsMap()

# Extract business latitude/longitude and transform them into sine and cosine values 
# to better represent spatial coordinates on a sphere. This often helps models to handle 
# spatial data more effectively than raw lat/long.
business_location_rdd = (business_rdd_base.map(lambda line: json.loads(line))  
    .filter(lambda x: x.get("latitude") is not None and x.get("longitude") is not None)  
    .map(lambda x: (
        x["business_id"],
        {"latitude_sin": math.sin(math.radians(x["latitude"])),
            "latitude_cos": math.cos(math.radians(x["latitude"])),
            "longitude_sin": math.sin(math.radians(x["longitude"])),
            "longitude_cos": math.cos(math.radians(x["longitude"]))})))

business_location = business_location_rdd.collectAsMap()



# ==============================
# Attribute Frequency Analysis
# ==============================

# attribute_key_counts = (business_rdd_base.map(json.loads)
#     .flatMap(lambda x: x['attributes'].keys() if x.get('attributes') else [])
#     .map(lambda key: (key, 1))
#     .reduceByKey(lambda a, b: a + b)
#     .collect())

# # sort the results by attribute keys
# sorted_attribute_key_counts = sorted(attribute_key_counts, key=lambda x: x[1])


# [('DietaryRestrictions', 138), ('Open24Hours', 352), ('RestaurantsCounterService', 397), ('AgesAllowed', 397), 
# ('Corkage', 657), ('BYOB', 911), ('BYOBCorkage', 1409), ('HairSpecializesIn', 1881), ('DriveThru', 6754), 
# ('BestNights', 6844), ('Smoking', 8113), ('CoatCheck', 8531), ('Music', 8807), ('GoodForDancing', 9162), 
# ('HappyHour', 9285), ('AcceptsInsurance', 11671), ('BusinessAcceptsBitcoin', 12674), ('DogsAllowed', 13681),
#  ('Caters', 40038), ('RestaurantsTableService', 43325), ('NoiseLevel', 43710), ('ByAppointmentOnly', 45423), 
# ('GoodForMeal', 47483), ('HasTV', 47533), ('Ambience', 47577), ('Alcohol', 47892), ('RestaurantsAttire', 48182), 
# ('WiFi', 49026), ('RestaurantsReservations', 51363), ('RestaurantsDelivery', 51668), ('WheelchairAccessible', 52023), 
# ('RestaurantsGoodForGroups', 53839), ('OutdoorSeating', 54181), ('RestaurantsTakeOut', 61206), ('GoodForKids', 64931), 
# ('BikeParking', 84891), ('BusinessParking', 103424), ('RestaurantsPriceRange2', 107120), ('BusinessAcceptsCreditCards', 140391)]


# From attribute frequency analysis, we have identified a set of common attributes 
# that are prevalent and potentially predictive. Here we map these attributes 
# (True/False) to numerical (1/0) values and store them in a dictionary.

common_business_attributes = (
    business_rdd_base
    .map(lambda x: json.loads(x)) 
    .map(lambda x: (x['business_id'],
        {"credit_card": 1 if x.get('attributes') and x['attributes'].get('BusinessAcceptsCreditCards') == "True"
                           else (0 if x.get('attributes') and x['attributes'].get('BusinessAcceptsCreditCards') == "False" else None),
            "price_range": x['attributes']['RestaurantsPriceRange2'] if x.get('attributes') and 'RestaurantsPriceRange2' in x['attributes'] else None,
            "bike_parking": 1 if x.get('attributes') and x['attributes'].get('BikeParking') == "True"
                            else (0 if x.get('attributes') and x['attributes'].get('BikeParking') == "False" else None),
            "kids_friendly": 1 if x.get('attributes') and x['attributes'].get('GoodForKids') == "True"
                             else (0 if x.get('attributes') and x['attributes'].get('GoodForKids') == "False" else None),
            "takeout": 1 if x.get('attributes') and x['attributes'].get('RestaurantsTakeOut') == "True"
                       else (0 if x.get('attributes') and x['attributes'].get('RestaurantsTakeOut') == "False" else None),
            "caters": 1 if x.get('attributes') and x['attributes'].get('Caters') == "True"
                      else (0 if x.get('attributes') and x['attributes'].get('Caters') == "False" else None),
            "table_service": 1 if x.get('attributes') and x['attributes'].get('RestaurantsTableService') == "True"
                             else (0 if x.get('attributes') and x['attributes'].get('RestaurantsTableService') == "False" else None),
            "by_appointment_only": 1 if x.get('attributes') and x['attributes'].get('ByAppointmentOnly') == "True"
                                   else (0 if x.get('attributes') and x['attributes'].get('ByAppointmentOnly') == "False" else None),
            "has_tv": 1 if x.get('attributes') and x['attributes'].get('HasTV') == "True"
                      else (0 if x.get('attributes') and x['attributes'].get('HasTV') == "False" else None),
            "reservations": 1 if x.get('attributes') and x['attributes'].get('RestaurantsReservations') == "True"
                            else (0 if x.get('attributes') and x['attributes'].get('RestaurantsReservations') == "False" else None),
            "delivery": 1 if x.get('attributes') and x['attributes'].get('RestaurantsDelivery') == "True"
                        else (0 if x.get('attributes') and x['attributes'].get('RestaurantsDelivery') == "False" else None),
            "wheelchair_accessible": 1 if x.get('attributes') and x['attributes'].get('WheelchairAccessible') == "True"
                                     else (0 if x.get('attributes') and x['attributes'].get('WheelchairAccessible') == "False" else None),
            "good_for_groups": 1 if x.get('attributes') and x['attributes'].get('RestaurantsGoodForGroups') == "True"
                               else (0 if x.get('attributes') and x['attributes'].get('RestaurantsGoodForGroups') == "False" else None),
            "outdoor_seating": 1 if x.get('attributes') and x['attributes'].get('OutdoorSeating') == "True"
                               else (0 if x.get('attributes') and x['attributes'].get('OutdoorSeating') == "False" else None),
            "dogs_allowed": 1 if x.get('attributes') and x['attributes'].get('DogsAllowed') == "True"
                            else (0 if x.get('attributes') and x['attributes'].get('DogsAllowed') == "False" else None),
            "happy_hour": 1 if x.get('attributes') and x['attributes'].get('HappyHour') == "True"
                          else (0 if x.get('attributes') and x['attributes'].get('HappyHour') == "False" else None),
            "good_for_dancing": 1 if x.get('attributes') and x['attributes'].get('GoodForDancing') == "True"
                                 else (0 if x.get('attributes') and x['attributes'].get('GoodForDancing') == "False" else None),
            "drive_thru": 1 if x.get('attributes') and x['attributes'].get('DriveThru') == "True"
                          else (0 if x.get('attributes') and x['attributes'].get('DriveThru') == "False" else None)}))).collectAsMap()


# ============================
# Attribute Encoding Analysis
# ============================

# attribute_key = "Smoking"

# possible_values = (business_rdd_base.map(json.loads)  
#     .filter(lambda x: x is not None and 'attributes' in x and x['attributes'] is not None)  
#     .map(lambda x: x['attributes'].get(attribute_key)) 
#     .filter(lambda x: x is not None)
#     .distinct() 
#     .collect())

# 'NoiseLevel': ['average', 'loud', 'quiet', 'very_loud']
# 'WiFi': ['no', 'free', 'paid']  
# 'RestaurantsAttire': ['casual', 'dressy', 'formal']
# 'Alcohol': ['none', 'full_bar', 'beer_and_wine']  
# 'Smoking': ['no', 'outdoor', 'yes']  


# Encoding categorical features like NoiseLevel, WiFi, etc. 
# After experimentation, only NoiseLevel was retained as it improved the model’s RMSE.
feature_encodings = {'NoiseLevel': {'average': 1, 'loud': 2, 'quiet': 0, 'very_loud': 3},
    'WiFi': {'no': 0, 'free': 1, 'paid': 2},
    'RestaurantsAttire': {'casual': 0, 'dressy': 1, 'formal': 2},
    'Alcohol': {'none':0, 'full_bar':1, 'beer_and_wine': 2},
    'Smoking': {'no': 0, 'outdoor': 1, 'yes': 2}}

encoded_features_rdd = (business_rdd_base.map(lambda x: json.loads(x))
    .filter(lambda x: x is not None)
    .map(lambda x: (
        x['business_id'],
        {"noise_level": feature_encodings['NoiseLevel'].get((x.get('attributes') if isinstance(x.get('attributes'), dict) else {}).get('NoiseLevel'), None),
            "wifi": feature_encodings['WiFi'].get((x.get('attributes') if isinstance(x.get('attributes'), dict) else {}).get('WiFi'), None),
            "restaurants_attire": feature_encodings['RestaurantsAttire'].get((x.get('attributes') if isinstance(x.get('attributes'), dict) else {}).get('RestaurantsAttire'), None),
            "alcohol": feature_encodings['Alcohol'].get((x.get('attributes') if isinstance(x.get('attributes'), dict) else {}).get('Alcohol'), None),
            "smoking": feature_encodings['Smoking'].get((x.get('attributes') if isinstance(x.get('attributes'), dict) else {}).get('Smoking'), None)})))

encoded_features = encoded_features_rdd.collectAsMap()

# ==============================
# Load and process check-in data
# ==============================

checkin_rdd = recommendation_system.textFile(checkin)

# Sum all check-ins for each business across all time keys. 
# Extract_number is used to parse the time segments if needed.
def extract_number(time_key):
    return int(re.search(r'\d+$', time_key).group())  

business_checkins = checkin_rdd.map(json.loads).map(lambda x: (x['business_id'], sum(count for time_key, count in x['time'].items() if (extract_number(time_key)))))
business_checkins_sum = business_checkins.collectAsMap()

# ==========================
# Load and process tip data
# ==========================

reviews_rdd = recommendation_system.textFile(tip)
reviews_rdd = reviews_rdd.map(json.loads).map(lambda x: ((x['user_id'], x['business_id']), x["likes"]))

# Calculate average likes per user and per business as fallback features.
review_likes_user_average = reviews_rdd.map(lambda x: (x[0][0], x[1])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()
review_likes_business_average = reviews_rdd.map(lambda x: (x[0][1], x[1])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()

reviews_rdd = reviews_rdd.collectAsMap()

# ============================
# Load and process photo data
# ============================

photo_rdd = recommendation_system.textFile(photo)
# Count how many photos are associated with each business.
business_photo_counts = photo_rdd.map(json.loads).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()


# ==============================
# Feature computation functions
# ==============================

def calculate_time_since(last_review_date, current_date=datetime.now()):
    # Calculate how long it has been since a user first started using Yelp.
    # This temporal feature may help model user maturity or experience level.
    if last_review_date:
        last_date = datetime.strptime(last_review_date, '%Y-%m-%d')  
        return (current_date - last_date).days
    else:
        return None
    
def create_train_dataset(user_business_rating):
    # Given a (user_id, business_id, rating) from the training set, build the feature vector (X) and label (y).
    user = user_business_rating[0]
    business = user_business_rating[1]
    rating = user_business_rating[2]

    # Extract user and business features, with fallback defaults if missing.
    if user in user_rdd and business in business_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        business_checkins_feature = business_checkins_sum.get(business, 0)
        business_photo_count = business_photo_counts.get(business, 0)
        review_likes = reviews_rdd.get((user, business), None)
        time_yelping = calculate_time_since(user_yelping_since)
    elif user in user_rdd:
        # If we have user info but not business info, fallback for business averages.
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = 3
        business_count = None
        business_checkins_feature = None
        business_photo_count = None
        review_likes = review_likes_user_average.get(user, None)
        time_yelping = calculate_time_since(user_yelping_since)
    elif business in business_rdd:
        # If we have business info but not user info, fallback for user averages.
        user_avg = 3
        user_count = None
        user_yelping_since = None
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        business_checkins_feature = business_checkins_sum.get(business, 0)
        business_photo_count = business_photo_counts.get(business, 0)
        review_likes = review_likes_business_average.get(business, None)
        time_yelping = None
    else:
        # If no info available, all set to None or default values.
        user_avg, user_count, business_avg, business_count, business_checkins_feature, review_likes, business_photo_count = None, None, None, None, None, None, None
        time_yelping = None

    # Get business attributes from the mapped dictionary.
    attrs = common_business_attributes.get(business, {})
    business_cc = attrs.get("credit_card", None)
    business_price_range = attrs.get("price_range", None)
    bike_parking = attrs.get("bike_parking", None)
    kids_friendly = attrs.get("kids_friendly", None)
    takeout = attrs.get("takeout", None)
    caters = attrs.get("caters", None)
    table_service = attrs.get("table_service", None)
    by_appointment_only = attrs.get("by_appointment_only", None)
    has_tv = attrs.get("has_tv", None)
    reservations = attrs.get("reservations", None)
    delivery = attrs.get("delivery", None)
    wheelchair_accessible = attrs.get("wheelchair_accessible", None)
    good_for_groups = attrs.get("good_for_groups", None)
    outdoor_seating = attrs.get("outdoor_seating", None)
    dogs_allowed = attrs.get("dogs_allowed", None)
    happy_hour = attrs.get("happy_hour", None)
    good_for_dancing = attrs.get("good_for_dancing", None)
    drive_thru = attrs.get("drive_thru", None)

    # User-level aggregated features (sum of interactions and compliments).
    user_sum = user_rdd_features.get(user, None)
    user_compliments = user_rdd_compliment.get(user, None)

    # Encoded categorical feature (NoiseLevel).
    noise = encoded_features.get(business, {}).get("noise_level", None)

    # Business open status and location-based sine/cosine features.
    business_open = business_is_open.get(business, None)
    bus_lat_sin = business_location.get(business, {}).get("latitude_sin", None)
    bus_lat_cos = business_location.get(business, {}).get("latitude_cos", None)
    bus_lon_sin = business_location.get(business, {}).get("longitude_sin", None)
    bus_lon_cos = business_location.get(business, {}).get("longitude_cos", None)

    # Construct the feature vector X. The first two fields (user, business) are identifiers 
    # and should not be fed into the model. The model will use the features starting from index 2 onwards.
    X = [user, business, user_avg, user_count, business_avg, business_count,
        business_checkins_feature, review_likes, business_photo_count, business_cc,
        business_price_range, bike_parking, kids_friendly, takeout, caters,
        table_service, by_appointment_only, has_tv, reservations, delivery,
        wheelchair_accessible, good_for_groups, outdoor_seating, dogs_allowed,
        happy_hour, good_for_dancing, drive_thru, user_sum, user_compliments, noise,
        business_open, bus_lat_sin, bus_lat_cos, bus_lon_sin, bus_lon_cos, time_yelping]
    y = rating
    return X, y

def create_test_dataset(user_business):
    # Similar to create_train_dataset, but no rating is provided.
    # We construct the same feature vector for the test set.
    user = user_business[0]
    business = user_business[1]

    if user in user_rdd and business in business_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        business_checkins_feature = business_checkins_sum.get(business, 0)
        business_photo_count = business_photo_counts.get(business, 0)
        review_likes = reviews_rdd.get((user, business), None)
        time_yelping = calculate_time_since(user_yelping_since)
    elif user in user_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = 3
        business_count = None
        business_checkins_feature = None
        business_photo_count = None
        review_likes = review_likes_user_average.get(user, None)
        time_yelping = calculate_time_since(user_yelping_since)
    elif business in business_rdd:
        user_avg = 3
        user_count = None
        user_yelping_since = None
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        business_checkins_feature = business_checkins_sum.get(business, 0)
        business_photo_count = business_photo_counts.get(business, 0)
        review_likes = review_likes_business_average.get(business, None)
        time_yelping = None
    else:
        user_avg, user_count, business_avg, business_count, business_checkins_feature, review_likes, business_photo_count = None, None, None, None, None, None, None
        time_yelping = None

    # Retrieve same set of attributes for test instances.
    attrs = common_business_attributes.get(business, {})
    business_cc = attrs.get("credit_card", None)
    business_price_range = attrs.get("price_range", None)
    bike_parking = attrs.get("bike_parking", None)
    kids_friendly = attrs.get("kids_friendly", None)
    takeout = attrs.get("takeout", None)
    caters = attrs.get("caters", None)
    table_service = attrs.get("table_service", None)
    by_appointment_only = attrs.get("by_appointment_only", None)
    has_tv = attrs.get("has_tv", None)
    reservations = attrs.get("reservations", None)
    delivery = attrs.get("delivery", None)
    wheelchair_accessible = attrs.get("wheelchair_accessible", None)
    good_for_groups = attrs.get("good_for_groups", None)
    outdoor_seating = attrs.get("outdoor_seating", None)
    dogs_allowed = attrs.get("dogs_allowed", None)
    happy_hour = attrs.get("happy_hour", None)
    good_for_dancing = attrs.get("good_for_dancing", None)
    drive_thru = attrs.get("drive_thru", None)

    user_sum = user_rdd_features.get(user, None)
    user_compliments = user_rdd_compliment.get(user, None)
    noise = encoded_features.get(business, {}).get("noise_level", None)
    business_open = business_is_open.get(business, None)
    bus_lat_sin = business_location.get(business, {}).get("latitude_sin", None)
    bus_lat_cos = business_location.get(business, {}).get("latitude_cos", None)
    bus_lon_sin = business_location.get(business, {}).get("longitude_sin", None)
    bus_lon_cos = business_location.get(business, {}).get("longitude_cos", None)

    X = [user, business, user_avg, user_count, business_avg, business_count,
        business_checkins_feature, review_likes, business_photo_count, business_cc,
        business_price_range, bike_parking, kids_friendly, takeout, caters,
        table_service, by_appointment_only, has_tv, reservations, delivery,
        wheelchair_accessible, good_for_groups, outdoor_seating, dogs_allowed,
        happy_hour, good_for_dancing, drive_thru, user_sum, user_compliments, noise,
        business_open, bus_lat_sin, bus_lat_cos, bus_lon_sin, bus_lon_cos, time_yelping]
    y = 0
    return X, y

# ======================================
# Prepare the training and test datasets
# ======================================

# Convert training RDD to arrays of (X, y).
train_sets = train_rdd.map(create_train_dataset).collect()
test_sets = test_rdd.map(create_test_dataset).collect()

# Extract features and labels for the training set.
# We exclude the first two fields (user_id, business_id) from the model input.
X_train = np.array([x[0][2:] for x in train_sets])  
y_train = np.array([x[1] for x in train_sets])

# Extract features for the test set.
X_test = np.array([x[0][2:] for x in test_sets]) 

# ==================
# Model Training
# ==================

# Use XGBoost Regressor with previously found optimal hyperparameters (e.g. via Optuna).
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',n_estimators=287,max_depth=7,
    learning_rate=0.05960638748789017,gamma=0.22700291558708435,
    colsample_bytree=0.6989551012623144,subsample=0.9768856652723699,
    reg_alpha=0.5906041463445368,reg_lambda=1.8975504313542997)

# Train the model on the training data.
xgb_model.fit(X_train, y_train)

# Make predictions on the test set.
predictions = xgb_model.predict(X_test)

# =======================================
# Write predictions to the output CSV file
# =======================================

with open(output, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id','business_id','prediction'])
    for i, row in enumerate(test_rdd.collect()):
        user_id, business_id = row
        writer.writerow([user_id,business_id,predictions[i]])


# ============================
# Additional functionalities like RMSE calculation and error distribution
# were tested locally and commented out in the final model.
# ============================

# Below commented code shows how RMSE and error distribution can be calculated if 
# ground truth labels are available.

# def calc_rmse(y_true, y_pred):
#     mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
#     rmse = math.sqrt(mse)
#     return rmse

# # Code for evaluating RMSE if we had y_test:
# real_y_rdd = recommendation_system.textFile(testing_set)
# head = real_y_rdd.first()
# y_test = real_y_rdd.filter(lambda line: line != head).map(lambda row: float(row.strip().split(',')[2])).collect()
# rmse = calc_rmse(y_test, predictions)
# print(f"Root Mean Square Error (RMSE): {rmse}")

# Distribution of errors:
# def count_levels(y_true, y_pred):
#     abs_diff = np.abs(np.array(y_true) - np.array(y_pred))
#     levels = {">=0 and <1": 0,
#               ">=1 and <2": 0,
#               ">=2 and <3": 0,
#               ">=3 and <4": 0,
#               ">=4": 0}
    
#     for diff in abs_diff:
#         if 0 <= diff < 1:
#             levels[">=0 and <1"] += 1
#         elif 1 <= diff < 2:
#             levels[">=1 and <2"] += 1
#         elif 2 <= diff < 3:
#             levels[">=2 and <3"] += 1
#         elif 3 <= diff < 4:
#             levels[">=3 and <4"] += 1
#         elif diff >= 4:
#             levels[">=4"] += 1
#     return levels

# levels_error = count_levels(y_test, predictions)
# for level, count in levels_error.items():
#     print(f"{level}: {count}")

# ============================
# Optuna hyperparameter tuning code snippet was done earlier and is commented out.
# This code is just for reference.
# ============================

# # Optuna code snippet for tuning hyperparameters:
# # The best hyperparameters found were used in the final model above.

############################# OPTUNA #############################

# # define the objective function for Optuna
# def objective(trial):
#     params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'gamma': trial.suggest_float('gamma', 0, 0.3),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
#         'objective': 'reg:squarederror', 
#         'eval_metric': 'rmse',         
#         'tree_method': 'auto'}

#     # Create and train the model
#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)

#     # Predict on the validation set
#     y_pred = model.predict(X_val_split)
#     rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))

#     return rmse

# create the Optuna study
# study = optuna.create_study(direction='minimize')  # We want to minimize RMSE
# study.optimize(objective, n_trials=50)  # Run 50 trials

# best parameters
# best_params = study.best_params
# print("Best parameters found by Optuna:", best_params)

# train the final model using the best parameters
# final_model = xgb.XGBRegressor(**best_params)
# final_model.fit(X_train_split, y_train_split)

# train predictions and RMSE
# train_predictions = final_model.predict(X_train_split)
# train_rmse = np.sqrt(mean_squared_error(y_train_split, train_predictions))
# print(f"Train RMSE: {train_rmse}")

# validation predictions and RMSE
# val_predictions = final_model.predict(X_val_split)
# val_rmse = np.sqrt(mean_squared_error(y_v

