import pandas as pd

df=pd.read_csv("C:\\Users\\Ruman Asif\\Documents\\Personal\\guvi\\"
               "final project\\3.1-data-sheet-udemy-courses.csv")

print("There are",len(df.columns),"columns which are:") #,df.columns)
for each in df.columns:
    print(each)


print("---------------------------------------------------")

print("Dropping columns which dont affect ratings of a user")
df.drop(['course_id','course_title','url','published_timestamp'],axis=1,inplace=True)
print("There are",len(df.columns),"columns left out which are:") #,df.columns)
for each in df.columns:
    print(each)

print("---------------Details---------------------------------")
print(df.info())

print("---------------------------------------------------")
print("number of nulls:",df.isnull().sum())

print("---------------------------------------------------")
print("There are just 4 null rows so dropping them as it wont"
      "affect our data significantly")
df.dropna(axis=0,inplace=True)
print("---------------------------------------------------")
print()
print()
print("Now the total number of nulls in our df are:",df.isnull().sum().sum())
print()
print("As the level and subject columns are text based, lets encode them")

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['level'] = label_encoder.fit_transform(df['level'])
df['subject'] = label_encoder.fit_transform(df['subject'])
print()
print("Let's have a look at out dataframe")
print()
print()
print(df.head().to_string())

# print(df.head(),df.tail())
print(df.tail().to_string())

print("-----------------------")

#------------------------------outlier starts here--------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(df['price'])
plt.title('Feature: Price: No outliers detected here')
plt.show()
sns.boxplot(df['num_lectures'])
plt.title('Feature: num_lectures: Upper outliers detected here')

plt.show()
sns.boxplot(df['content_duration'])
plt.title('Feature: content_duration: outliers detected '
          'here but we choose to keep them as course duration can change as is requirement')

plt.show()

import numpy as np
Q1=np.percentile(df['num_lectures'],25)
Q3=np.percentile(df['num_lectures'],75)
IQR=Q3-Q1
upper= df['num_lectures']>=(Q3+1.5*IQR)
upper_outliers_index=np.where(upper)
lower= df['num_lectures']<=(Q1-1.5*IQR)
print("lower bound bound:")
lower_outliers_index=np.where(lower)
#--------------------------------------dropping outliers-----------------------
print("--------------------dropping upper outliers from no. of lectures now-------------------")
for each in upper_outliers_index:
    df.drop(each,inplace=True)

#-------------------------collinearity starts here---------------------
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print()
print()
print()
print("Now checking for collinearity")
print()

X = df[['price', 'num_subscribers', 'num_reviews','num_lectures','level',
        'content_duration','subject']]

X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

print("As we can see from the the table the features are not collinear as "
      "none of the features have VIF greater than 5. So keeping all these feature.......")
print()
print(".....for now.....")

#---------------------trying xgboost-----------------------------

print("Let's try xgboost for our model")
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

X = df[['price', 'num_subscribers', 'num_reviews','num_lectures','level',
        'content_duration','subject']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2) #, random_state=42)

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
print()
print("As our MSE is low let's try other ways")
print()
print()

#----------------------------------------------------------------------

# level and subject dropped here-------------
df.drop(['level','subject'],axis=1,inplace=True) # dropping level and subject also here

print("let's try dropping level and subject and see if our model fits better")
print()
print()

#------------------------------trying knn------------------------------

print("let's try K-nn this time:")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['price', 'num_subscribers', 'num_reviews','num_lectures',
        'content_duration']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
y_pred = knn_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error in k-nn is:", mse)
print("Hmmm... not that good.... ")
print("lets try somehing else....")

# --------------------num lectures dropped here-------------

df.drop(['num_lectures'],axis=1,inplace=True)

#------------------------------trying knn------------------------------

print("let's try k-nn now")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X = df[['price', 'num_subscribers', 'num_reviews',
        'content_duration']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_regressor = KNeighborsRegressor(n_neighbors=5)

knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("K-nn Mean Squared Error:", mse)
print("knn r2 score",r2_score(y_test, y_pred))
print()
print()

#---------------------multiple linear here--------------

print("let's try multiple linear now...")
print()
print()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
X = df[['price', 'num_subscribers', 'num_reviews',
        'content_duration']]
y = df['Rating']
# X=X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print("regression score in multiple linear now is:",regr.score(X_test, y_test))
print()
print()

#---------------trying xgboost here------------------
print("lets try xgboost again now as we have dropped number of lectures....")

import xgboost as xgb

from sklearn.model_selection import train_test_split

X = df[['price', 'num_subscribers', 'num_reviews',
        'content_duration']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2) #, random_state=42)

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

print()

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error in xgboost now:", mse)
print()
print()
print()
print("Hmmm..  not good at all... Can this model predict at all?")
print("--------------------------------------------------------------------------------------")




