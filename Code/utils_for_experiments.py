import datetime
import os
import logging
import subprocess

### ---------------------------------------------------------------- ###

# This file contains utility functions for experiments, e.g. creating 
# directories, setting up logs or running commands.

### ---------------------------------------------------------------- ###

def format_time(current_time):
    current_time = current_time.replace(":", "_").split(".")[0]
    return current_time

def create_dir(dir_path, format='%Y-%m-%d %H:%M:%S'):
    current_time = datetime.datetime.now()
    if format=='isoformat':
        current_time = current_time.isoformat()
    else:
        current_time = current_time.strftime(format)
    current_time = format_time(current_time)
    results_output_path = os.path.join(dir_path, current_time)
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(results_output_path, exist_ok=True)
    return results_output_path

def setup_logging(log_path, log_name):
    os.makedirs(log_path, exist_ok=True)
    log_path = os.path.join(log_path, log_name)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

def save_params(params, save_path, filename):
    file = os.path.join(save_path, filename)
    with open(file, "w") as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')

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
        logging.error(f"Errors (if any):\n {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")