#%%
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import pandas as pd


data = load_breast_cancer()

X = pd.DataFrame(data=data.data,columns=data.feature_names)
y = pd.DataFrame(data=data.target,columns=['target'])


#%% Spli the data
from sklearn.model_selection import train_test_split

splits = train_test_split(X,y, test_size=0.2)
train_X,test_X = splits[0],splits[1]
train_y,test_y = splits[2],splits[3]

# lets do standard scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_train = scaler.fit_transform(train_X)
scale_test = scaler.transform(test_X)

#%%

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape = (scale_train.shape[1],)),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')                      
    ])


model.compile(optimizer = 'adam',metrics=['accuracy',tf.keras.metrics.FalsePositives()],loss='binary_crossentropy')
history = model.fit(scale_train,train_y,epochs=100)

#%%
test_loss, test_accuracy ,fp= model.evaluate(scale_test, test_y)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Total :{scale_test.shape[0]} , False positive :{fp}")
# Make predictions
predictions = model.predict(test_X)
binary_predictions = (predictions > 0.5).astype(int)
