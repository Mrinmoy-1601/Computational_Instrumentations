def calculate_forward_difference(input_data):
    n = len(input_data)
    forward_diff_table = [input_data]

    for i in range(1, n):
        prev_row = forward_diff_table[-1]
        next_row = [prev_row[j + 1] - prev_row[j] for j in range(len(prev_row) - 1)]
        forward_diff_table.append(next_row)

    return forward_diff_table
def print_forward_difference_table(forward_diff_table):
    for row in forward_diff_table:
        print(" ".join(map(str, row)))
