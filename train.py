import numpy as np
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.random.rand(100,1)
y = 3*X + np.random.randn(100,1)*0.1

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)

mse = mean_squared_error(y,pred)

metrics = {"mse": float(mse)}

with open("metrics.json","w") as f:
    json.dump(metrics,f)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model Trained")
print("MSE:",mse)

print("Hubert Mathew Sunil - 2022BCS0052")