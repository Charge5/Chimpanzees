'''
This is the script used to generate the A-GEM data of the figure of appendix C.
This script is required to generate the data for the script "hyperparameters_impact_on_forgetting.py"
'''

import subprocess
from itertools import product
import time
from pathlib import Path
import os

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

    learning_rates = [0.001]
    depth = [10]
    width = [5]
    epochs = [25,50,75,100]
    seed = 0

    # Generate Cartesian product
    combinations = list(product(learning_rates, epochs, depth, width))
    total_combinations = len(combinations)

    for i, (lr, e, d, w) in enumerate(combinations, start=1):
        print(f"Iteration {i}/{total_combinations} - Learning rate: {lr}, Epochs: {e}, Depth: {d}, Width: {w}, Seed: {seed}")
        start_time = time.time()  # Start timing
        main_path = os.path.join(Path(__file__).resolve().parent,"src/mammoth/utils/main.py")
        cmd = f"python {main_path} --dataset seq-mnist --backbone mnistmlp --model agem --buffer_size 500 --lr {lr} --seed {seed} --n_epochs {e} --mlp_hidden_size {w} --mlp_hidden_depth {d} --enable_other_metrics True"
        run_command(cmd)
        end_time = time.time()  # Start timing
        elapsed_time = end_time - start_time
        print(f"Time for iteration {i}: {elapsed_time:.4f} seconds\n")

