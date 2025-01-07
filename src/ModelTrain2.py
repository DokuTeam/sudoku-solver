import numpy as np
import tensorflow as tf


# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the data: Normalize the images and reshape them for CNN input
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0  # Reshape to (28, 28, 1) and normalize
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0    # Reshape to (28, 28, 1) and normalize
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encode labels
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Data Augmentation: Apply augmentation techniques to improve model generalization
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

datagen.fit(x_train)

# 4. Build the CNN model for digit classification
model = tf.keras.models.Sequential()

# Convolutional layers with ReLU activation and MaxPooling layers
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and apply dense layers for classification
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout to prevent overfitting
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 classes for digits 0-9

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Implement early stopping to avoid overfitting and save the best model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 7. Train the model using augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# 8. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 9. Save the trained model
model.save('resources/digit_recognition_model.h5')
print("Model saved as 'digit_recognition_model.h5'")

# 10. Optional: Make predictions with the trained model
# For example, predicting the digit for a new image
# digit_img should be preprocessed to (1, 28, 28, 1) before feeding it to the model
# pred = model.predict(digit_img)
# print(f"Predicted digit: {np.argmax(pred)}")