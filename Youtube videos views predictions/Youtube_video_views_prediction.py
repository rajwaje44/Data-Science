
# importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import dataset
# filtering was first done in excel by normally observing the data
data = pd.read_csv("D:/dataforpython/hackathon_2/train_1.csv")
data1 = pd.DataFrame.copy(data)   # so that original data will not be affected and will do all process on data1.

# to see na values
data1.isnull().sum()
data.info()
data1["publish_date"] = pd.to_datetime(data1["publish_date"])
data1["trending_date"] = pd.to_datetime(data1["trending_date"])

data1.info()

# deriving a new variable "day" which will have diffrence between publish date and trending date
from datetime import date
d0 = data1["publish_date"].dt.date
d1 = data1["trending_date"].dt.date
delta = d1 - d0
data1["day"] = delta
data1.info()            # day column is in time format we need to convert it into string

# converting data to object type
data1["day"] = data1["day"].astype(str)
data1["day"] = data1["day"].map(lambda x: x.strip("")[0])
data1["day"] = data1["day"].astype(int)
data1.info()

# replacing blanks with na
data1 = data1.replace([" "],np.nan)
data1["comment_count"] = data1["comment_count"].fillna(0)


# removing obs with views less than 100
data2 = data1.loc[~data1["views"].str.contains("#")]
data2["views"] = data2["views"].astype(int)
data2 = data2.loc[data2["views"]>100]
data2 = data2.loc[data2['likes'] <= 25000]
data2 = data2.loc[data2['Trend_day_count'] <15]
data2 = data2.loc[data2['Tag_count'] < 26]
data2 = data2.loc[data2["category_id"]<40]
data2.info()

# value_counts will show how much time each category is repeated
data2["category_id"].value_counts()

data2.describe([0.75,0.85,0.95])

# now we have cleaned data except for subscribers which has some NA values
# dropping na values
data3 = data2.dropna()
data3.info()

# converting variables to interger format
data3["subscriber"] = data3["subscriber"].astype("int64")
data3["comment_count"] = data3["comment_count"].astype("int64")
data3["comment_disabled"] = data3["comment_disabled"].astype("int64")
data3["like dislike disabled"] = data3["like dislike disabled"].astype("int64")
data3["tag appered in title"] = data3["tag appered in title"].astype("int64")
data3["views"] = data3["views"].astype("int64")
data3.info()

# setting x and y i.e independent and dependent variables
x = data3[['category_id','subscriber', 'Trend_day_count','Tag_count', 'Trend_tag_count', 'comment_count', 'likes', 'dislike','comment_disabled','tag appered in title','like dislike disabled']]
y = data3[['views']]

# log tranformation
import numpy as np
y_log  = np.log(y)
x_log  = np.log1p(x)

# feature scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_log)
x_log = scaler.transform(x_log)
print(x_log)

# BELOW CODE IS TO SEE HOW MODEL PERFORMS ON OLS MODEL
"""#
applying ols model
import statsmodels.api as sm
reg = sm.OLS(y_log, x_log)
result = reg.fit()
result.summary()


# calculating vif


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df["features"] = x.columns
vif_df["VIF Factor"] = [vif(x.values, i) for i in range(x.shape[1])]
vif_df.round(2)
"""

# data preprocessing test data
test = pd.read_csv("D:/dataforpython/hackathon_2/test.csv")

# replacing blanks with na
test = test.replace([" "],np.nan)
test["comment_count"] = test["comment_count"].fillna(0)
test["category_id"].value_counts()
test.info()

# FILLING NA VALUES WITH MEDIAN VALUE
for a in test.columns[:]:
    if test[a].dtype == 'float64':
        test[a].fillna(test[a].median(), inplace=True)

test["publish_date"] = pd.to_datetime(test["publish_date"])
test["trending_date"] = pd.to_datetime(test["trending_date"])

test.info()

# deriving a new variable "day" which will have diffrence between publish date and trending date
from datetime import date
e0 = test["publish_date"].dt.date
e1 = test["trending_date"].dt.date
delta1 = e1 - e0
test["day"] = delta1
test.info()

test["day"] = test["day"].astype(str)
test["day"] = test["day"].map(lambda x: x.strip(" ")[0])
test["day"] = test["day"].astype(int)
test.info()


test.isnull().sum()
test = test[['category_id','subscriber', 'Trend_day_count','Tag_count', 'Trend_tag_count', 'comment_count', 'likes', 'dislike','comment_disabled','tag appered in title','like dislike disabled']]

# CONVERT COLUMNS WITH BOOL TO INT FORMAT
test["subscriber"] = test["subscriber"].astype("int64")
test["comment_count"] = test["comment_count"].astype("int64")
test["comment_disabled"] = test["comment_disabled"].astype("int64")
test["like dislike disabled"] = test["like dislike disabled"].astype("int64")
test["tag appered in title"] = test["tag appered in title"].astype("int64")
test.info()

# standard scale for test data
x_log_test = np.log1p(test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_log_test)
x_log_test1 = scaler.transform(x_log_test)


####
# xgboost MODEL
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100,gamma=0.75)
xgb.fit(x_log, y_log)
y_pred_xgb = xgb.predict(x_log_test1)
y_pred_xgb1 = np.exp(y_pred_xgb)
prediction = pd.DataFrame(y_pred_xgb1,np.arange(1,1336))
prediction.to_csv("D:/dataforpython/hackathon_2/submit15.csv")


























