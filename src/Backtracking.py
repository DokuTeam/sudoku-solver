import numpy as np

def is_valid(board, row, col, num):
    if num in board[row, :] or num in board[:, col]:
        return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row + 3, start_col:start_col + 3]:
        return False

    return True

def select_unassigned_variable(board):
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                return row, col
    return None

def backtrack(board):
    empty_cell = select_unassigned_variable(board)
    if not empty_cell:
        return True

    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(board, row, col, num):

            board[row, col] = num

            if backtrack(board):
                return True

            board[row, col] = 0

    return False

def solve_sudoku(board):
    if backtrack(board):
        return board
    else:
        return None

def display_sudoku(label, sudoku):
    print(label)

    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                print('.', end='')
            else:
                print(cell, end='')
            if (j + 1) % 3 == 0 and j < 8:
                print(' |', end='')

            if j != 8:
                print('  ', end='')
        print('\n', end='')
        if (i + 1) % 3 == 0 and i < 8:
            print("--------+----------+---------")

def main():

    n = 1
    display_input_output = True

    quizzes = np.zeros((n, 81), np.int32)
    solutions = np.zeros((n, 81), np.int32)

    for i, line in enumerate(open('../resources/sudoku.csv', 'r').read().splitlines()[1:n + 1]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s

    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))

    for i in range(0, n):
        quiz, solution = quizzes[i], solutions[i]

        if display_input_output:
            display_sudoku('\nPuzzle:', quiz)

        output = solve_sudoku(quiz)
        is_output_correct = np.array_equal(solution, output)

        if not is_output_correct:
            raise Exception("Output is not same as solution.")

        if display_input_output:
            display_sudoku('\nOutput:', output)
            display_sudoku('\nSolution:', solution)

if __name__ == '__main__':
    main()
