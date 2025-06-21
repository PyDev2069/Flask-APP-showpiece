import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("homes.csv")

df = pd.DataFrame(data)

X=df[['Area','Bedroom']]
y=df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

with open("home.pkl","wb") as f:
    pickle.dump(model,f)



