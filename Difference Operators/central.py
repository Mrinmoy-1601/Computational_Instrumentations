def calculate_central_difference(input_data):
    n = len(input_data)
    central_diff_table = [input_data]

    for i in range(1, n):
        prev_row = central_diff_table[-1]
        next_row = [prev_row[j + 1] - prev_row[j] for j in range(len(prev_row) - 1)]
        central_diff_table.append(next_row)

    return central_diff_table
def print_central_difference_table(central_diff_table):
    for row in central_diff_table:
        print(" ".join(map(str, row)))