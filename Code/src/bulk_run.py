import subprocess
from itertools import product
import time

def run_command(command: str):
    """
    Run a command-line command and print the output.

    :param command: The command to execute as a string.
    """
    try:
        # Execute the command and capture the output
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            check=True
        )
        # Print the command's output
        # print("Output:\n", result.stdout)
        print("Errors (if any):\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Error output:\n", e.stderr)

# Example usage
if __name__ == "__main__":
    print("Start multiple run...")

    learning_rates = [0.001,0.01]
    # learning_rates = [0.01]
    epochs = [10,20]
    depth = [2,5,10]
    # width = [x for x in range(25,25,5)]
    # width = [50,75]
    width = [100]
    # Generate Cartesian product
    combinations = list(product(learning_rates, epochs, depth, width))
    total_combinations = len(combinations)

    for i, (lr, ep, d, w) in enumerate(combinations, start=1):
        print(f"Iteration {i}/{total_combinations} - Learning rate: {lr}, Epochs: {ep}, Depth: {d}, Width: {w}")
        start_time = time.time()  # Start timing
        cmd = f"python mammoth/utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf-mc  --lr {lr} --seed 42 --n_epochs {ep} --mlp_hidden_size {w} --mlp_hidden_depth {d}"  # Replace with your desired command
        run_command(cmd)
        end_time = time.time()  # Start timing
        elapsed_time = end_time - start_time
        print(f"Time for iteration {i}: {elapsed_time:.4f} seconds\n")

    #
    # for lr, ep, d, w in product(learning_rates, epochs, depth, width):
    #     print(f"Learning rate: {lr}, Epochs: {ep}, Depth: {d}, Width: {w}")
    #     cmd = f"python mammoth/utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf-mc  --lr {lr} --seed 42 --n_epochs {ep} --mlp_hidden_size {w} --mlp_hidden_depth {d}"  # Replace with your desired command
    #     #     run_command(cmd)
    #
    #
