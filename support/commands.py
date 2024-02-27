import subprocess
import re


def extract_numbers(input_string):
    return re.findall(r'\d+', input_string)

def get_gpu_availablity():
    
    cmd_command = 'nvidia-smi'
    process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        return False
    else:
        output = stdout.decode()
        output.split('MiB')
        available_memory = int(extract_numbers(output.split('MiB')[1])[0])
        if available_memory > 2000:
            return True
        else:
            return False