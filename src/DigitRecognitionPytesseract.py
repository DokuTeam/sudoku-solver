import cv2
import numpy as np
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract'  # For macOS/Linux

class DigitRecognizer:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        """
        Preprocess the image for better OCR performance.
        - Convert to grayscale (only if it's not already grayscale)
        - Threshold the image to binarize
        - Resize image if necessary
        """
        if len(image.shape) == 3:  # 3 channels (BGR image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply a more robust thresholding method
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # cv2.imshow("Preprocessed Image", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Resize image to improve OCR accuracy
        resized = cv2.resize(thresh, (320, 320))  # Adjust size as needed

        return resized

    def recognize_digit(self, image):
        """
        Recognize digits in the provided image using OCR.
        """
        preprocessed_image = self.preprocess_image(image)

        custom_config = r'--oem 3 --psm 10'

        try:
            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
        except Exception as e:
            return 0

        if not text.strip():  # If OCR returns empty string
            return 0

        # Filter out non-digit characters and get the recognized digits
        digits = ''.join(filter(str.isdigit, text))

        if not digits:  # If no digits are found
            return 0

        return digits

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