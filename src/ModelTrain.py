import tensorflow as tf


def load_and_preprocess_data():
    """
    Load and preprocess the mnist dataset.

    Returns:
        tuple: Processed training and test data (X_train, y_train, X_test, y_test).
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape to include a channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

def build_model():
    """
    Build a Convolutional Neural Network (CNN) model for digit recognition.

    Returns:
        keras.Model: The compiled model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax")  # 10 classes for digits 0-9
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(model, X_train, y_train, X_test, y_test, model_path):
    """
    Train the model on the training data and save it to a file.

    Args:
        model (keras.Model): The compiled model.
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
        model_path (str): Path to save the trained model.
    """
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Build the model
    print("Building the model...")
    model = build_model()

    # Train and save the model
    MODEL_PATH = "../resources/digit_recognition_model.h5"
    print("Training the model...")
    train_and_save_model(model, X_train, y_train, X_test, y_test, MODEL_PATH)
