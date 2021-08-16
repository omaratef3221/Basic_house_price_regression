import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("USA_Housing.csv")
print("Data types: ", data.dtypes)
print("Data Statistics: ", data.describe())
data.drop('Address', inplace = True, axis=1)
print("Missing values:", data.isnull().sum())
x = data.iloc[:,0:-1]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
model = LinearRegression()
model.fit(x_train,y_train)
predicted_result = model.predict(x_test)
print("MSE: ",mean_squared_error(y_test,predicted_result))
print("R2: ",r2_score(y_test,predicted_result))