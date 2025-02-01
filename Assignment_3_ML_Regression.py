import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis,zscore,boxcox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression #pip install scikit_learn to install sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR


#fetching the data from the source
hou= fetch_california_housing()
houdb=pd.DataFrame(hou.data,columns=hou.feature_names)


# checking the shape and other details
print(houdb.head())
print(houdb.tail())
print(houdb.shape)
print(houdb.info())
print(houdb.describe())



#checked for duplicate and null values
print("NUll values"+str(houdb.isnull().sum()))
print("Duplicate values"+str(houdb.duplicated().sum()))


#differentiate the data iinto numerical and non numerical
numerical= houdb.select_dtypes(include=['number']).columns
nonnumerical=houdb.select_dtypes(include=['bool','object']).columns
houdbnum = houdb[numerical]
houdbchar=houdb[nonnumerical]
print("Shape of categorical column is"+str(houdbchar.shape))



#checking th skewness of the data using skew
# houdbnum=houdbnum.skew()
# print("First Skewness"+str(houdbnum))


#histplot to check the skewness
houdb.hist(bins=30)
plt.xlabel("Histogram")
plt.show()


#using iqr to reduce the skewness


#reducing the skewness 

#avebdrms outlier removal
iqr1=houdbnum['AveBedrms'].quantile(0.25)
iqr3=houdbnum['AveBedrms'].quantile(0.75)
iqr= iqr3-iqr1

lowerwisk = iqr1 - iqr * 1.5  # Correct: Lower bound
upperwisk = iqr3 + iqr * 1.5  # Correct: Upper bound
houdbnum=houdbnum[(houdbnum['AveBedrms'] >= lowerwisk) & (houdbnum['AveBedrms'] <= upperwisk)]






houdbnum['AveRooms']=np.log(houdbnum['AveRooms'])
houdbnum['Population']=np.log(houdbnum['Population'])
houdbnum['AveBedrms']=np.log(houdbnum['AveBedrms'])
houdbnum['AveOccup']=np.log(houdbnum['AveOccup'])




houdbnum_af_trans=houdbnum.skew()


print("Second Skewness"+str(houdbnum_af_trans))
plt.boxplot(houdbnum)
plt.show()



#checking the correlation
corel= houdbnum.corr()
print(corel)


#heatmap
sns.heatmap(corel,annot=True, cmap="coolwarm")
plt.show()


#perform onehot encoding or label encoding  if you have categorical column

#for reference and for future
# now perform minmax scaler

#     Standardization is often the default choice if your model assumes that the data is normally distributed or if you're working with algorithms sensitive to the variance in features.
#     MinMax Scaling should be used if you specifically need features to be in a defined range or if you're working with algorithms like neural networks that perform better with normalized data.

# Which is better?

#     There is no one-size-fits-all answer. 
# It depends on your dataset and the type of model you're using.
#  If you're unsure, you can test both methods and compare performance on your model.





#perfomring minmax scaler

datascale2= pd.DataFrame(houdbnum)
minscaler=MinMaxScaler()
minscaler.fit(datascale2)
minscaled=minscaler.transform(datascale2)
print(minscaled)





#differentitation X an Y  and spliting data to train and test

X=houdbnum[['MedInc','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]
y = houdbnum['HouseAge']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Xtrain shape{X_train.shape} Xtrain test{X_test.shape}Ytrain shape{y_train.shape}Ytrain Test{y_test.shape}")

################################################################################################################
# linear Regression model
housemodel = LinearRegression()
housemodel.fit(X_train,y_train)
print(housemodel)
y_pred= housemodel.predict(X_test)
print(y_pred)
results= pd.DataFrame({'Actual Values': y_test,'Predicted Values': y_pred})
print("Linear prediction"+str(results))

#chekcing the acquracy score
print("MAE:", mean_absolute_error(y_test, y_pred))
##################################################################################################################\

################################################################################################################
# Deciaion tree regressor
#checing the best combination for regressor
parameter={
    # 'criterion':['gini','entropy'], #spliing qulaity metrics
    'max_depth':[1,2,3,4,5],#how deep tree can grow ,no of level tree can have upto here its 5
    'max_features':['log2','sqrt'] #how many feature model will look 

}
modelregr= DecisionTreeRegressor(criterion='squared_error')
modelregr.fit(X_train,y_train)
predictedDTR=modelregr.predict(X_test)
results1= pd.DataFrame({'Actual Values': y_test,'Predicted Values': predictedDTR})
print("Decisiontree prediction"+str(results1))

###############################################Random Forest Regression

modelrfr = RandomForestRegressor()
modelrfr.fit(X_train,y_train)
predictrfr=modelrfr.predict(X_test)
results_rf = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': predictrfr})
print("Random Forest Regressor prediction\n" + str(results_rf))

############################################### Gradien Boosting Regression

modelgbr = GradientBoostingRegressor()
modelgbr.fit(X_train,y_train)
predictgbr=modelgbr.predict(X_test)
results_gbr = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': predictgbr})
print("Gradien Boosting Regression\n" + str(results_gbr))


############################################### SVR

modelsvr = SVR()
modelsvr.fit(X_train,y_train)
predictsvr=modelsvr.predict(X_test)
results_svr = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': predictsvr})
print("SVR \n" + str(results_svr))



# Evaluation of the model
print("Dataset Accuracy of Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Evaluation of the model
print("Dataset Accuracy of Decision tree")
print("MAE:", mean_absolute_error(y_test, predictedDTR))
print("MSE:", mean_squared_error(y_test, predictedDTR))
print("R² Score:", r2_score(y_test, predictedDTR))

# Evaluation of the model
print("Dataset Accuracy of Random Forest")
print("MAE:", mean_absolute_error(y_test, predictrfr))
print("MSE:", mean_squared_error(y_test, predictrfr))
print("R² Score:", r2_score(y_test, predictrfr))

# Evaluation of the model
print("Dataset Accuracy of Gradient Boosting")
print("MAE:", mean_absolute_error(y_test, predictgbr))
print("MSE:", mean_squared_error(y_test, predictgbr))
print("R² Score:", r2_score(y_test, predictgbr))

# Evaluation of the model
print("Dataset Accuracy of SVR")
print("MAE:", mean_absolute_error(y_test, predictsvr))
print("MSE:", mean_squared_error(y_test, predictsvr))
print("R² Score:", r2_score(y_test, predictsvr))



















# cv=GridSearchCV(modelregr,parameter,scoring=['neg_mean_squared_error'],refit='neg_mean_squared_error')
# print(cv.best_estimator_)





##################################################################################################################\
































