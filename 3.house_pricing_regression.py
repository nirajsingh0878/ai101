#%%
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
housing = fetch_california_housing()
X,y = pd.DataFrame(data=housing.data,columns=housing.feature_names),pd.DataFrame(data=housing.target,columns=['target'])

# traintest split
splits = train_test_split(X,y,test_size=0.2)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(splits[0])
test_X = scaler.transform(splits[1])
train_y = splits[2]
test_y = splits[3]

#%%
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape=(train_X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation='linear')
])
model.compile(optimizer='adam',loss='mae',metrics=['mae','mse'])
history = model.fit(train_X,train_y,batch_size=32,validation_split=0.2,epochs=100)
# %%
import numpy as np
test_loss, test_mae, test_mse = model.evaluate(test_X, test_y)
print(f"\nTest MAE: ${test_mae * 100000:.2f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: ${np.sqrt(test_mse) * 100000:.2f}")
# %%
predictions = model.predict(test_X[:10])
print("\nSample Predictions vs Actual Prices (in $100,000s):")
for i in range(10):
    print(f"Predicted: ${predictions[i][0]:.2f}, Actual: ${test_y.iloc[i,0]:.2f}")
