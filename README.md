# NYC Airbnb Price Prediction

### Project Description
Machine learning project that predicts NYC Airbnb prices using advanced feature engineering, geospatial clustering, full EDA, preprocessing, and a tuned XGBoost model. Includes model diagnostics and business insights. Final model performs strongly on log-price prediction.

### Dataset
- **Source:** Inside Airbnb – NYC 2019  
- **Rows:** ~48,000  
- **Target:** `log_price` (transformed from `price`)  
- **Main Features:** room type, neighbourhood, latitude/longitude, reviews, availability, host activity  

### Feature Engineering
- `neighborhood_average_price` — mean price per neighbourhood  
- `location_cluster` — KMeans geospatial clusters  
- `has_reviews` — binary review indicator  
- `stay_category` — derived from minimum_nights (short/medium/long stay)  

### Preprocessing
- Median imputation for numeric features  
- Most frequent imputation for categorical features  
- One-Hot Encoding → 200+ encoded features  
- Train/test split: **80/20**  

### Model
- **Algorithm:** XGBoost Regressor  
- **Hyperparameter Search:** RandomizedSearchCV (50 trials, 3-fold CV)  
- **Metrics:** MAE, RMSE, R² (log scale + original scale)

### Model Performance
**Log Scale (trained target = log_price):**  
- **MAE:** 0.2898  
- **RMSE:** 0.3899  
- **R²:** 0.6588  

**Original Price Scale:**  
- **MAE:** \$44.70  
- **RMSE:** \$87.30  
- **R²:** 0.4537  

### Top Features (XGBoost)
- `room_type_Entire home/apt`  
- `neighbourhood_group_Manhattan`  
- `neighborhood_average_price`  
- `latitude`, `longitude`  
- `location_cluster`  
- `calculated_host_listings_count`  

### Key Insights
- Entire homes/apartments drive the highest prices  
- Manhattan dominates pricing across all boroughs  
- Geospatial clustering significantly improves model accuracy  
- Hosts with more listings tend to price higher  
- Model underestimates very high-priced listings (rare in dataset)

### Dependencies
- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn  
- xgboost  
