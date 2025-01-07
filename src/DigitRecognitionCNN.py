import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self, model_path):
        """
        Initialize the DigitRecognizer with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained Keras model.
        """
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_digit(self, digit_img):
        """
        Preprocess the digit image for model input.

        Args:
            digit_img (numpy.ndarray): The digit image to preprocess.

        Returns:
            numpy.ndarray: The preprocessed image.
        """

        # for mnist_resnet_model.h5
        # resized = cv2.resize(digit_img, (32, 32))
        # normalized = resized / 255.0
        # rgb_image = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
        # preprocessed = rgb_image.reshape(1, 32, 32, 3)

        # for digit_recognition_model.h5
        resized = cv2.resize(digit_img, (28, 28))
        normalized = resized / 255.0
        preprocessed = normalized.reshape(1, 28, 28, 1)

        # image_to_show = preprocessed.reshape(28, 28)
        # plt.imshow(image_to_show, cmap='gray')
        # plt.title("Preprocessed Digit")
        # plt.axis("off")  # Hide axes
        # plt.show()

        return preprocessed

    def recognize_digit(self, digit_img):
        """
        Recognize a single digit from an image.

        Args:
            digit_img (numpy.ndarray): The digit image to recognize.

        Returns:
            int: The recognized digit (0-9), or -1 if the digit is not clear.
        """
        preprocessed = self.preprocess_digit(digit_img)
        prediction = self.model.predict(preprocessed)
        # Get the digit with the highest probability
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit]

        # Optional: Define a confidence threshold
        if confidence > 0.8:
            return predicted_digit
        else:
            return 0  # Indicate unclear digit

    def process_grid(self, grid_images):
        """
        Process a grid of digit images.

        Args:
            grid_images (list of numpy.ndarray): 2D list of digit images.

        Returns:
            numpy.ndarray: 2D NumPy array of recognized digits (0-9), with 0 indicating an empty cell.
        """
        sudoku_grid = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                if grid_images[i][j] is not None:  # Non-empty cell
                    digit = self.recognize_digit(grid_images[i][j])
                    sudoku_grid[i][j] = digit if digit != -1 else 0
        return sudoku_grid
