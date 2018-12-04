# Dataquest_Predict_Stock_Market
Solution of the project Predicting stock market(Dataquest)
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("sphist.csv")



data["Date"] = pd.to_datetime(data["Date"] )

#print(data[data["Date"] > datetime(year=2015, month=4, day=1)].shape)

data = data.sort_values(by = 'Date',ascending=True)

data["mean_past_5d"] = 0

data["mean_past_30d"] = 0

data["mean_past_365d"] = 0



data["mean_past_5d"] = data["mean_past_5d"].rolling(5).mean()
data["mean_past_5d"] = data["mean_past_5d"].shift()
data["mean_past_30d"] = data["mean_past_30d"].rolling(30).mean()
data["mean_past_30d"] = data["mean_past_30d"].shift()
data["mean_past_365d"] = data["mean_past_365d"].rolling(365).mean()
data["mean_past_365d"] = data["mean_past_365d"].shift()

#print(data.head(5))

#print(data[data["Date"] < datetime(year=1951, month=1, day=3)].shape)
data = data.drop(data[data["Date"] < datetime(year=1951, month=1, day=3)].index)

#print(data.shape)
data = data.dropna(axis = 0)
#print(data.shape)

train = data[data["Date"] < datetime(year=2013, month=1, day=1)]

test = data[data["Date"] >= datetime(year=2013, month=1, day=1)]


lr1 = LinearRegression()
lr1.fit(train[['mean_past_5d']],train[['Close']])
predictions1 = lr1.predict(test[['mean_past_5d']])
mae_5 = mean_absolute_error(test['Close'],predictions1)

print(mae_5)


lr2 = LinearRegression()
lr2.fit(train[['mean_past_30d']],train[['Close']])
predictions2 = lr2.predict(test[['mean_past_30d']])
mae_30 = mean_absolute_error(test['Close'],predictions2)

print(mae_30)

lr3 = LinearRegression()
lr3.fit(train[['mean_past_5d','mean_past_30d','mean_past_365d']],train[['Close']])
predictions3 = lr3.predict(test[['mean_past_5d','mean_past_30d','mean_past_365d']])
mae_365 = mean_absolute_error(test['Close'],predictions3)

print(mae_365)

lr3 = LinearRegression()
lr3.fit(train[['mean_past_365d']],train[['Close']])
predictions3 = lr3.predict(test[['mean_past_365d']])
mae_365 = mean_absolute_error(test['Close'],predictions3)

print(mae_365)

