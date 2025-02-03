import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, boxcox, zscore
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Load dataset
bcanc = load_breast_cancer()
target = pd.Series(bcanc.target)
bcanc = pd.DataFrame(bcanc.data, columns=bcanc.feature_names)
bcanc['target'] = target

# Basic info
print(bcanc.info())
print(bcanc.describe())
print(bcanc.shape)
print(bcanc.head())

# Check for null and duplicate values
print("Null values: " + str(bcanc.isnull().sum()))
print("Duplicate values: " + str(bcanc.duplicated().sum()))

# Separate numerical and non-numerical columns
numerical = bcanc.select_dtypes(include=['number']).columns
nonnumerical = bcanc.select_dtypes(include=['bool', 'object']).columns
bcancnum = bcanc[numerical]
bcancchar = bcanc[nonnumerical]

print("Shape of categorical columns: " + str(bcancchar.shape))
print("Skewness before handling: " + str(bcancnum.skew()))

# Handling outliers using IQR
for col in bcancnum.select_dtypes(include=['float64']).columns:  # Only apply to numerical columns
    # Calculate the first (Q1) and third (Q3) quartiles
    iqr1 = bcancnum[col].quantile(0.25)
    iqr3 = bcancnum[col].quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    iqr = iqr3 - iqr1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = iqr1 - (1.5 * iqr)
    upper_bound = iqr3 + (1.5 * iqr)
    
    # Filter the DataFrame to keep only the rows within the bounds
    bcancnum = bcancnum[(bcancnum[col] >= lower_bound) & (bcancnum[col] <= upper_bound)]

# Apply transformations to reduce skewness
# We do not apply transformations on the target variable
# bcancnum = zscore(bcancnum)  # Normalize the numerical features

print("Skewness after handling: " + str(pd.DataFrame(bcancnum).skew()))

# Boxplot to check the effect of transformations
plt.boxplot(bcancnum)
plt.show()

# MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bcancnum)

# Prepare data for machine learning
X = pd.DataFrame(scaled_data, columns=bcancnum.columns)
y = target  # Keep target separate and unchanged

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Logistic Regression
bcanclr = LogisticRegression(max_iter=10000)  # Increased max_iter for convergence
bcanclr.fit(X_train, y_train)
print(bcanclr)
y_predlr = bcanclr.predict(X_test)
print(y_predlr)
resultslr = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_predlr})
print("Logistic Regression Prediction Results:")
print(resultslr)

# 2. Decision Tree Classifier
bcanctree = DecisionTreeClassifier(random_state=42)
bcanctree.fit(X_train, y_train)
y_preddt = bcanctree.predict(X_test)
resultsdt = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_preddt})

print("\nDecision Tree Classifier Prediction Results:")
print(resultsdt)
print(classification_report(y_test, y_preddt))

# 3. Random Forest Classifier
bcanrf = RandomForestClassifier(n_estimators=100, random_state=42)
bcanrf.fit(X_train, y_train)
y_predrf = bcanrf.predict(X_test)
resultsrf = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_predrf})

print("\nRandom Forest Classifier Prediction Results:")
print(resultsrf)
print(classification_report(y_test, y_predrf))

# 4. Support Vector Machine (SVM)
bcansvm = SVC(kernel='linear')
bcansvm.fit(X_train, y_train)
y_predsvm = bcansvm.predict(X_test)
resultssvm = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_predsvm})

print("\nSupport Vector Machine (SVM) Prediction Results:")
print(resultssvm)
print(classification_report(y_test, y_predsvm))

# 5. k-Nearest Neighbors (k-NN)
bcanknn = KNeighborsClassifier(n_neighbors=5)
bcanknn.fit(X_train, y_train)
y_predknn = bcanknn.predict(X_test)
resultsknn = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': y_predknn})

print("\nK-Nearest Neighbors (k-NN) Prediction Results:")
print(resultsknn)
print(classification_report(y_test, y_predknn))