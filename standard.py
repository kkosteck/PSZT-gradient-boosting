# Import models and utility functions 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn import datasets 
  
# Setting SEED for reproducibility 
SEED = 1
  
# Importing the dataset  
housing = datasets.load_boston() 
X, y = housing.data, housing.target 
  
# Splitting dataset 
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = SEED) 
  
# Instantiate Gradient Boosting Regressor 
gbr = GradientBoostingRegressor(n_estimators = 200, max_depth = 4, random_state = SEED) 
  
# Fit to training set 
gbr.fit(train_X, train_y) 
  
# Predict on test set 
pred_y = gbr.predict(test_X) 
  
# test set RMSE 
test_rmse = MSE(test_y, pred_y) ** (1 / 2) 
  
# Print rmse 
print('RMSE test set: {:.2f}'.format(test_rmse)) 