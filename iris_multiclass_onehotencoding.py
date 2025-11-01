#%%
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 

# TODO :one hot encoding 

iris = load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = pd.DataFrame(iris.target,columns=['target'])
splits = train_test_split(X,y,test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(splits[0])
X_test = scaler.transform(splits[1])

#%%
import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']
)

#%%
history = model.fit(X_train,splits[2],epochs=100,
                    validation_split=0.2)


test_loss, test_accuracy = model.evaluate(X_test, splits[3])
print(f"Test Accuracy: {test_accuracy:.4f}")

