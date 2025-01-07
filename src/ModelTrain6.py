import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


# ### loading mist hand written dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# ## Applying threshold for removing noise
_,X_train_th = cv2.threshold(X_train,127,255, cv2.THRESH_BINARY)
_,X_test_th = cv2.threshold(X_test,127,255, cv2.THRESH_BINARY)



# ### Reshaping
X_train = X_train_th.reshape(-1,28,28,1)
X_test = X_test_th.reshape(-1,28,28,1)


# ### Creating categorical output from 0 to 9
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)


# ## cross checking shape of input and output
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # Creating CNN model

input_shape = (28,28,1)
number_of_classes = 10

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,epochs=5, shuffle=True,
                    batch_size = 200,validation_data= (X_test, y_test))


model.save('digit_classifier2.h5')