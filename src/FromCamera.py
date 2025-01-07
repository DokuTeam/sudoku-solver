import cv2
import numpy as np
from BoardDetection import BoardDetector
from DigitRecognitionPytesseract import DigitRecognizer
from Backtracking import solve_sudoku

# Helper function to overlay solution on the image
def overlay_solution(image, grid, solution):
    cell_height = image.shape[0] // 9
    cell_width = image.shape[1] // 9
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:  # Only overlay on empty cells
                x = j * cell_width + cell_width // 3
                y = i * cell_height + 2 * cell_height // 3
                cv2.putText(image, str(solution[i][j]), (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Initialize the Board Detector and Digit Recognizer
board_detector = BoardDetector()
digit_recognizer = DigitRecognizer()

# Open webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detect Sudoku board
        largest_contour = board_detector.detect_board(frame)
        if largest_contour is not None:
            board_image = board_detector.extract_board(frame, largest_contour)

            # Split the board into a grid
            grid = board_detector.split_grid(board_image)

            # Recognize digits and create the Sudoku grid
            sudoku_grid = []
            for row in grid:
                sudoku_row = []
                for cell in row:
                    digit = digit_recognizer.recognize_digit(cell)
                    sudoku_row.append(digit)
                sudoku_grid.append(sudoku_row)

            # Solve the Sudoku puzzle
            solution = [row[:] for row in sudoku_grid]  # Create a copy to solve
            if solve_sudoku(solution):
                # Overlay the solution on the original board
                frame_with_solution = overlay_solution(frame, sudoku_grid, solution)
                cv2.imshow("Sudoku Solver", frame_with_solution)
            else:
                cv2.putText(frame, "Sudoku Not Solvable", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Sudoku Solver", frame)

        else:
            cv2.putText(frame, "No Sudoku Board Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Sudoku Solver", frame)

    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Sudoku Solver", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
