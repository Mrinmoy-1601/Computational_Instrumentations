from backward import calculate_backward_difference, print_backward_difference_table
from forward import calculate_forward_difference, print_forward_difference_table
from central import calculate_central_difference, print_central_difference_table

def main():
    input_data = list(map(int, input().split()))
    print("Choose your difference operator\n")
    print("[f] Forward Difference\n [b] Backward Difference\n [c] Central Difference")
    print("Enter your choice: ")
    choice = input().lower()
    if choice=='f':
        diff_table = calculate_forward_difference(input_data)
        print_forward_difference_table(diff_table)
    elif choice=='b':
        diff_table = calculate_backward_difference(input_data)
        print_backward_difference_table(diff_table)
    elif choice=='c':
        diff_table = calculate_central_difference(input_data)
        print_central_difference_table(diff_table)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()