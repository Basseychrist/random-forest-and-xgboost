# Random Forest and XGBoost Model Comparison

This lab compares the performance of Random Forest and XGBoost regression models for predicting house prices using the California Housing Dataset.

## Libraries and Imports

```python
import numpy as np                          # Numerical computing library
import matplotlib.pyplot as plt              # Data visualization library
from sklearn.datasets import fetch_california_housing  # Load California Housing dataset
from sklearn.model_selection import train_test_split   # Split data into train/test sets
from sklearn.ensemble import RandomForestRegressor     # Random Forest regression model
from xgboost import XGBRegressor            # XGBoost regression model
from sklearn.metrics import mean_squared_error, r2_score  # Performance metrics
import time                                 # Measure execution time
```

## Code Explanation

### Data Loading and Preprocessing

```python
data = fetch_california_housing()  # Load the California Housing dataset
X, y = data.data, data.target      # Extract features (X) and target values (y)
```
- **fetch_california_housing()**: Downloads the California Housing dataset containing 20,640 observations with 8 features
- **X**: Feature matrix containing housing attributes (MedInc, HouseAge, AveRooms, etc.)
- **y**: Target values representing median house prices

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **train_test_split()**: Splits data into 80% training and 20% testing sets
- **test_size=0.2**: Allocates 20% of data for testing
- **random_state=42**: Ensures reproducible splits across different runs

### Model Initialization

```python
n_estimators = 100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)
```
- **n_estimators**: Number of trees in the ensemble (100 trees for each model)
- **RandomForestRegressor()**: Creates a Random Forest model that trains multiple decision trees independently
- **XGBRegressor()**: Creates an XGBoost model that trains trees sequentially, with each tree correcting previous errors
- **random_state=42**: Ensures reproducible results

### Model Training with Timing

```python
start_time_rf = time.time()        # Record start time
rf.fit(X_train, y_train)           # Train Random Forest model
end_time_rf = time.time()          # Record end time
rf_train_time = end_time_rf - start_time_rf  # Calculate training duration
```
- **time.time()**: Returns current time in seconds
- **fit()**: Trains the model on training data
- **rf_train_time**: Elapsed time in seconds for training the Random Forest

```python
start_time_xgb = time.time()        # Record start time
xgb.fit(X_train, y_train)           # Train XGBoost model
end_time_xgb = time.time()          # Record end time
xgb_train_time = end_time_xgb - start_time_xgb  # Calculate training duration
```
- Same as above, but for XGBoost model

### Making Predictions with Timing

```python
start_time_rf = time.time()        # Record start time
y_pred_rf = rf.predict(X_test)     # Generate predictions on test set
end_time_rf = time.time()          # Record end time
rf_pred_time = end_time_rf - start_time_rf  # Calculate prediction duration
```
- **predict()**: Uses trained model to make predictions on new data
- **y_pred_rf**: Array of predicted house prices from Random Forest
- **rf_pred_time**: Time taken to make all predictions

```python
start_time_xgb = time.time()        # Record start time
y_pred_xgb = xgb.predict(X_test)     # Generate predictions on test set
end_time_xgb = time.time()          # Record end time
xgb_pred_time = end_time_xgb - start_time_xgb  # Calculate prediction duration
```
- Same as above, but for XGBoost model

### Performance Metrics

```python
mse_rf = mean_squared_error(y_test, y_pred_rf)  # Calculate Mean Squared Error
r2_rf = r2_score(y_test, y_pred_rf)             # Calculate R-squared score
```
- **mean_squared_error()**: Measures average squared difference between predicted and actual values (lower is better)
  - Formula: MSE = (1/n) * Σ(actual - predicted)²
- **r2_score()**: Measures proportion of variance explained by model (closer to 1.0 is better)
  - Formula: R² = 1 - (SS_res / SS_tot)

```python
mse_xgb = mean_squared_error(y_test, y_pred_xgb)  # Calculate Mean Squared Error
r2_xgb = r2_score(y_test, y_pred_xgb)             # Calculate R-squared score
```
- Same as above, but for XGBoost model

### Standard Deviation Calculation

```python
std_y = np.std(y_test)  # Calculate standard deviation of test target values
```
- **np.std()**: Computes standard deviation, used to show prediction confidence bands in visualization

### Visualization

```python
plt.figure(figsize=(14, 6))  # Create figure with size 14x6 inches
plt.subplot(1, 2, 1)         # Create 1x2 grid, select first subplot
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue", ec='k')  # Plot scatter points
```
- **plt.figure()**: Creates new plotting window
- **plt.subplot()**: Divides figure into grid and selects specific cell
- **plt.scatter()**: Creates scatter plot showing actual vs predicted values
  - **alpha=0.5**: Sets transparency level
  - **ec='k'**: Sets edge color to black

```python
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="perfect model")
```
- Plots a diagonal line representing perfect predictions
- **'k--'**: Black dashed line
- **lw=2**: Line width of 2 points

```python
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
```
- Plots upper and lower confidence bands (±1 standard deviation from perfect line)
- **'r--'**: Red dashed line
- Shows acceptable prediction range

```python
plt.subplot(1, 2, 2)         # Select second subplot
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="green", ec='k')  # Plot scatter points
```
- Same as first subplot, but for XGBoost model

```python
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
```
- Same as first subplot, adds perfect model and confidence band lines

```python
plt.legend()  # Show legend for labels
plt.show()    # Display the plots
```
- **legend()**: Displays legend explaining line colors
- **show()**: Renders the plots

## Key Differences: Random Forest vs XGBoost

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| **Training Method** | Parallel (independent trees) | Sequential (corrects previous errors) |
| **Training Speed** | Slower | Faster |
| **Prediction Speed** | Slower | Faster |
| **Accuracy** | Good | Often better |
| **Overfitting Risk** | Lower | Higher (requires tuning) |

## Performance Metrics Explained

- **MSE (Mean Squared Error)**: Average of squared prediction errors. Lower values indicate better predictions.
- **R² Score**: Proportion of variance explained by model. Ranges from 0 to 1, with 1.0 being perfect.
- **Training Time**: Time required to fit the model on training data.
- **Prediction Time**: Time required to generate predictions on test set.

## Observations from Results

1. XGBoost typically achieves lower MSE and higher R² than Random Forest
2. XGBoost is significantly faster in both training and prediction
3. Random Forest respects data bounds (doesn't exceed max/min values)
4. XGBoost may overshoot bounds but often with better overall accuracy
