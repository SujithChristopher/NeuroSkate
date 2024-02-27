import subprocess
import re

def get_gpu_availablity():
    cmd_command = 'nvidia-smi'
    try:
        output = subprocess.check_output(cmd_command, shell=True, stderr=subprocess.STDOUT).decode()
        match = re.search(r'(?<=MiB\s)\d+', output)
        print(re.search(r'MiB', output))
        if match:
            available_memory = int(match.group())
            return available_memory > 2000
        else:
            return False
    except subprocess.CalledProcessError:
        return False

get_gpu_availablity()