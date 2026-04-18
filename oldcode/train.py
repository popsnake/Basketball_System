import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1 = np.array(())
dataset2 = np.array(())
dataset3 = np.array(())
dataset4 = np.array(())

dataset5=dataset1.copy()
dataset5=dataset5+20

dataset6=dataset1.copy()
dataset6=dataset6+5

dataset9=dataset1.copy()
dataset9=dataset9-15

dataset10=dataset1.copy()
dataset10=dataset10+10



dataset7 = np.vstack((dataset1,dataset5,dataset6,dataset10,dataset9))
dataset8 = np.hstack((dataset2,dataset2,dataset2,dataset2,dataset2))

print(dataset1.shape)
print(dataset2.shape)
print(dataset7.shape)
print(dataset8.shape)

kernel_regularizer=keras.regularizers.l2(0.01)
keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.LSTM(units=128,return_sequences=True,input_shape=(31,75)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64),
    keras.layers.Dropout(0.5),
    keras.layers.LSTM(units=32,return_sequences=False),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5,activation='softmax'),




])

model.summary()

model.compile(
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

train_images=dataset1.copy()
train_lable=dataset2.copy()
train_images=dataset7.copy()
train_lable=dataset8.copy()
test_images=dataset3.copy()
test_lable=dataset4.copy()
print(dataset7)

train_images /= 1500

model.fit(train_images,train_lable,batch_size=15,epochs=250)

test_images /= 1500
test_loss, test_acc = model.evaluate(test_images,test_lable,verbose=2)

print('\nTest accuracy:', test_acc)
model.save(r'model_data/model.h5')

test_images /= 1500
test_loss, test_acc = model.evaluate(test_images,test_lable,verbose=2)

print('\nTest accuracy:', test_acc)

from google.colab import drive
drive.mount('/gdrive')
model.save('model/model.h5')
import shutil
shutil.copy('/content/model_data/model.h5','destination')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
np.argmax(predictions[6])