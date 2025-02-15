import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,zscore,kurtosis,boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif,SelectKBest,f_regression
from sklearn.model_selection import GridSearchCV

cardata = pd.read_csv('ML/CarPrice_Assignment.csv') #loading the dataset using pandas read_csv extension ,can also be done using pandas dataframe if the data is not table
print(cardata.info())
print(cardata.describe()) #Reffer:Data will not show  if print function is not give
print("Total Null Values")
print(cardata.isnull().sum())
print("Total Duplicate values")
print(cardata.duplicated().sum())
print(cardata.shape)

cardia=cardata.select_dtypes(include=['number'])
cardiastd=np.std(cardia)
cardia.hist(bins=30)
plt.xlabel("Histogram")
# plt.show()
print(cardia.skew())

# doing  log to reduce the skewness
cardia['price']=np.log(cardia['price'])
cardia['enginesize']=np.log(cardia['enginesize'])
cardia['wheelbase']=np.log(cardia['wheelbase'])
cardia['compressionratio'],_=boxcox(cardia['compressionratio'])
cardia['horsepower']=np.log(cardia['horsepower'])


#sekwness for the above has been done using log and boxcox.
# Tried log multiple time but value was not reducing so tried boxcox

print("After Handling the skewness")
print(cardia.skew())

# Now setting the data for prediction by diving the data to X and Y
print(cardia.shape)
X=cardia.drop(columns=['price'])
print(X.shape)
y=cardia['price']
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Xtrain shape{X_train.shape} Xtrain test{X_test.shape}Ytrain shape{y_train.shape}Ytrain Test{y_test.shape}")


# Now assigning the data to alogorithams to create models

# Linear Regression
carmodellinear = LinearRegression()
carmodellinear.fit(X_train,y_train)
print(carmodellinear)
y_pred= carmodellinear.predict(X_test)
print(y_pred)
results= pd.DataFrame({'Actual Values': y_test,'Predicted Values': y_pred})
print("Linear prediction"+str(results))

# Checking the performance of the model
print("Dataset Accuracy of Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))



# Decision Tree Regression
carmodeldec = DecisionTreeRegressor()
carmodeldec.fit(X_train,y_train)
print(carmodeldec)
y_pred= carmodeldec.predict(X_test)
print(y_pred)
results= pd.DataFrame({'Actual Values': y_test,'Predicted Values': y_pred})
print("Deciosintree prediction"+str(results))


print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("Rsquared:",r2_score(y_test,y_pred))


# Random Forest Regressor 
carmodelrand = RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=50)
carmodelrand.fit(X_train,y_train)
print(carmodelrand)
y_pred= carmodelrand.predict(X_test)
print(y_pred)
results= pd.DataFrame({'Actual Values': y_test,'Predicted Values': y_pred})
print("RandomForest prediction"+str(results))

print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("Rsquared:",r2_score(y_test,y_pred))

#Gradient Boosting Regressor

carmodelgrad= GradientBoostingRegressor()
carmodelgrad.fit(X_train,y_train)
print(carmodelgrad)
y_pred=carmodelgrad.predict(X_test)
print(y_pred)
results= pd.DataFrame({'Actual Values':y_test,'Predicted Values':y_pred})
print("Gradient Boosting"+str(results))

print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("Rsquared:",r2_score(y_test,y_pred))


#Support Vector Regressor

carmodelsvr= SVR()
carmodelsvr.fit(X_train,y_train)#setting  the data for training 
print(carmodelsvr)
y_pred=carmodelsvr.predict(X_test)#setting the data for testing 
print(y_pred)
results=pd.DataFrame({'Actual values':y_test,'Predicted Values':y_pred})
print("SVR"+str(results))

print("MSE:",mean_squared_error(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
print("Rsquared:",r2_score(y_test,y_pred))



#feature importance analysis
f_selector=SelectKBest(f_regression,k=10)
x_f=f_selector.fit_transform(cardia,y)
f_selected=cardia.columns[f_selector.get_support()]
print("Anova test:",f_selected)


# hyper tunning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],      # Maximum depth of trees
    'min_samples_split': [2, 5, 10]   # Minimum samples to split a node
}

# Perform Grid Search with Cross-Validation
rf_regressor = RandomForestRegressor()
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train,y_train)
print("Hyper Tunning",grid_search.best_params_)





















