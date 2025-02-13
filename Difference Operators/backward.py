def calculate_backward_difference(input_data):
    n = len(input_data)
    backward_diff_table = [input_data]

    for i in range(1, n):
        prev_row = backward_diff_table[-1]
        next_row = [prev_row[j] - prev_row[j-1] for j in range(1, len(prev_row))]
        backward_diff_table.append(next_row)

    return backward_diff_table
def print_backward_difference_table(backward_diff_table):
    for row in backward_diff_table:
        print(" ".join(map(str, row)))