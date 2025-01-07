import cv2
import numpy as np
import matplotlib.pyplot as plt

class BoardDetector:
    def __init__(self):
        pass

    def preprocess_image(self, image_path):
        """
        Preprocess the input image to prepare it for board detection.

        Args:
            image_path (str): Path to the input Sudoku image.

        Returns:
            numpy.ndarray: The preprocessed binary image.
        """
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary

    def find_largest_contour(self, binary_image):
        """
        Find the largest contour in the binary image, which should correspond to the Sudoku board.

        Args:
            binary_image (numpy.ndarray): Binary preprocessed image.

        Returns:
            numpy.ndarray: The largest contour.
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def extract_board(self, image, contour):
        """
        Extract the Sudoku board from the image using the largest contour.

        Args:
            image (numpy.ndarray): Original grayscale image.
            contour (numpy.ndarray): The largest contour.

        Returns:
            numpy.ndarray: The extracted board as a warped image.
        """
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4:
            raise ValueError("Unable to detect a rectangular Sudoku board.")

        # Define the points for perspective transformation
        points = approx.reshape(4, 2)
        points = sorted(points, key=lambda x: (x[1], x[0]))  # Sort by y, then x
        if points[0][0] > points[1][0]:
            points[0], points[1] = points[1], points[0]
        if points[2][0] < points[3][0]:
            points[2], points[3] = points[3], points[2]

        # Convert points to NumPy array
        points = np.array([points[0], points[1], points[2], points[3]], dtype='float32')

        top_left, top_right, bottom_right, bottom_left = points
        width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
        height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))

        return warped

    def split_grid(self, board_image):
        """
        Split the extracted Sudoku board into a 9x9 grid of individual cells.

        Args:
            board_image (numpy.ndarray): Extracted Sudoku board image.

        Returns:
            list of numpy.ndarray: 2D list of cell images.
        """
        height, width = board_image.shape
        cell_height, cell_width = height // 9, width // 9
        grid = []

        for i in range(9):
            row = []
            for j in range(9):
                y1, y2 = i * cell_height, (i + 1) * cell_height
                x1, x2 = j * cell_width, (j + 1) * cell_width
                cell = board_image[y1:y2, x1:x2]

                # Optional: Apply erosion to clean small artifacts
                cell = cv2.erode(cell, np.ones((1, 1), np.uint8))

                # threshold_value = 150  # Threshold value
                # max_value = 255  # Value to set for pixels above the threshold
                # _, cell = cv2.threshold(cell, threshold_value, max_value, cv2.THRESH_BINARY)

                # Crop a few pixels from each side to remove black edges
                margin = 2  # Number of pixels to crop from each side
                cell = cell[margin:-margin, margin:-margin]

                cell = cv2.bitwise_not(cell)

                row.append(cell)

            grid.append(row)

        return grid

    def detect_board(self, image_path):
        """
        Detect the Sudoku board, extract it, and split it into a grid of cells.

        Args:
            image_path (str): Path to the input Sudoku image.

        Returns:
            list of numpy.ndarray: 2D list of cell images.
        """
        preprocessed = self.preprocess_image(image_path)
        largest_contour = self.find_largest_contour(preprocessed)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        board_image = self.extract_board(image, largest_contour)
        grid = self.split_grid(board_image)

        fig, axes = plt.subplots(9, 9, figsize=(10, 10))
        fig.suptitle("Extracted Sudoku Grid Cells", fontsize=16)

        for i in range(9):
            for j in range(9):
                # Display each cell in the corresponding subplot
                axes[i, j].imshow(grid[i][j], cmap='gray')
                axes[i, j].axis("off")  # Turn off axes for better visualization

        plt.tight_layout()
        plt.show()

        return grid
