import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

def recognize_digit(image):
    result = reader.readtext(image, allowlist='0123456789')

    if len(result) > 0:
        return result[0][1]
    else:
        return 0


def process_grid(grid_images):
    sudoku_grid = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            if grid_images[i][j] is not None:  # Non-empty cell
                digit = recognize_digit(grid_images[i][j])
                sudoku_grid[i][j] = digit
    return sudoku_grid
