import pandas as pd
import numpy as  np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew,kurtosis,zscore
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler

employedb= pd.read_csv("ML/Employee(2).csv")
print(employedb.info())
print(employedb.describe())
print(employedb[['Age','Salary']].count())
employedb=employedb.rename(columns={"Age":"EmpAge"}) #renaming of column name
print(employedb.info())
numerical=employedb.select_dtypes(include=['number']).columns    #sorting numerical values
non_numerical=employedb.select_dtypes(include=['object','bool']).columns #sorting non-numerical values


x_num = employedb[numerical]
fornumscaling=x_num
x_cat=employedb[non_numerical]
forcatscaling=x_cat
print("Duplicates_nonnumerical"+str(x_num.duplicated().sum())) #counting total duplicate values
print("Dupicates numericals"+str(x_cat.duplicated().sum()))

x_num=x_num.drop_duplicates() #droping duplicate values
x_cat=x_cat.drop_duplicates()

print("Afternum"+str(x_num.duplicated().sum()))
print("Afternonnum"+str(x_cat.duplicated().sum()))
###### duplicate complete


####### checking null values started

x_cat['Company']=x_cat['Company'].fillna(x_cat['Company'].mode()[0])
x_cat['Place']=x_cat['Place'].fillna(x_cat['Place'].mode()[0])
x_num['EmpAge']=x_num['EmpAge'].fillna(x_num['EmpAge'].mean())
x_num['Salary']=x_num['Salary'].fillna(x_num['Salary'].median())

print("Null value count")
print(x_num.isnull().sum())
print("NUll values non number")
print(x_cat.isnull().sum())

# dropduplicatesnum['Age']=dropduplicatesnum['Age'].replace(0,np.nan)

#scaling the skewed values
print("Total Skeweness")
x_num['Gender']=np.sqrt(x_num['Gender'])
x_num['EmpAge']=np.log(x_num['Salary'])
print(x_num.skew())
print(x_num.isna().sum())  # For DataFrame
print(x_num.isnull().sum())  

x_cat['Company'].value_counts().plot(kind='bar')
plt.xlabel("Categorical column")
# plt.show()



plt.boxplot(x_num)
plt.xlabel("NUmerical values")
plt.show()
print("Before sorting"+str(x_num.shape))
x_num = x_num[(x_num['EmpAge'] <40) & (x_num['Salary'] < 5000)]
x_num = x_num.sort_values(by='Salary')
print("After sorting"+str(x_num.shape))
plt.plot(x_num['Salary'],x_num['EmpAge'],linestyle='--',color='red',linewidth=2)
plt.show()

#one hot encoding
cat_encod=pd.get_dummies(data=x_cat,columns=["Company","Place","Country"])
print(cat_encod)

x_cat_cols=["Company","Place","Country"]
new_encoded_list=[]
for encode in x_cat_cols:
    new_encoded_list += [f"is_{category}" for category in x_cat[encode].unique().tolist()] 
print(new_encoded_list)
onehot=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
onehot1=onehot.fit_transform(x_cat[['Company','Place','Country']])
print(onehot1)
x_catencode=pd.DataFrame(onehot1,columns=new_encoded_list)
finalencoding=x_cat.join(x_catencode)
print("Encoded Values")
print(finalencoding)

#label encoding

label_encoder = LabelEncoder()
x_cat_cols=x_cat[["Company","Place","Country"]]
for i in x_cat_cols:
    x_cat_cols[i] = label_encoder.fit_transform(x_cat_cols[i])
print(x_cat[["Company","Place","Country"]].value_counts())


#standardscaler perform only if you havent done no other kind of transformation or scaling

datascale= pd.DataFrame(fornumscaling)
scaler = StandardScaler()
scaler.fit(datascale)
scaled_data=scaler.transform(datascale)
print(scaled_data)

#minmax sclaer

datascale2= pd.DataFrame(fornumscaling)
minscaler=MinMaxScaler()
minscaler.fit(datascale2)
minscaled=minscaler.transform(datascale2)
print(minscaled)






