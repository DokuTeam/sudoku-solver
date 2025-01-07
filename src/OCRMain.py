import numpy as np

import Backtracking
from BoardDetection import BoardDetector
from DigitRecognitionEasyocr import process_grid


if __name__ == "__main__":
    IMAGE_PATH = "../resources/images/test-5.png"

    board_detector = BoardDetector()

    try:
        # Step 1: Detect the Sudoku board and split into grid cells
        print("Detecting and extracting Sudoku board...")
        grid_images = board_detector.detect_board(IMAGE_PATH)

        # Step 2: Recognize digits in the grid
        print("Recognizing digits in the Sudoku grid...")

        sudoku_grid = process_grid(grid_images)

        Backtracking.display_sudoku("Initial Sudoku Grid:", np.array(sudoku_grid))

        # Step 3: Solve the Sudoku puzzle
        print("Solving the Sudoku puzzle...")
        result = Backtracking.solve_sudoku(sudoku_grid)

        if result.any():
            Backtracking.display_sudoku("Solved Sudoku Grid:", sudoku_grid)
        else:
            print("No solution found for the Sudoku puzzle.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
