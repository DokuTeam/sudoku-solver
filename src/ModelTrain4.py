import numpy as np
import tensorflow as tf
import cv2

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
ResNet50 = tf.keras.applications.ResNet50
mnist = tf.keras.datasets.mnist
to_categorical = tf.keras.utils.to_categorical
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Input = tf.keras.layers.Input

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: Normalize the images and reshape them for CNN input
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

# Resize the images from (28, 28) to (32, 32)
x_train = np.array([cv2.resize(img.squeeze(), (32, 32)) for img in x_train])
x_test = np.array([cv2.resize(img.squeeze(), (32, 32)) for img in x_test])

# Expand the dimensions to match (32, 32, 3) since ResNet50 expects 3 channels
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert to 3 channels (RGB) by repeating the single channel across 3 channels
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    shear_range=0.3,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Build the model with a pretrained ResNet50 base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
base_model.trainable = False

# Create the final model
model = Sequential([
    base_model,  # Pretrained ResNet50 model as base
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Train the model with early stopping and learning rate reduction
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, reduce_lr])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Fine-tune the model by unfreezing some of the base model layers
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all layers except the last 10
    layer.trainable = False

# Recompile the model after unfreezing layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training with the fine-tuned layers
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, reduce_lr])

# Evaluate the fine-tuned model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Fine-tuned Test accuracy: {test_acc}")

# Save the trained model
model.save("resources/mnist_resnet_model.h5")
print("Model saved as 'mnist_resnet_model.h5'")