import subprocess

def query_ollama(prompt, model):
    '''
    Queries ollama with given prompt and model
    model: the model to query (llama3 or wizard-vicuna-uncensored:13b)
    '''
    cmd = ['ollama', 'run', model]
    
    # Start the process
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Send the prompt to the process
    stdout, stderr = process.communicate(prompt)

    # Check for errors
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None

    return stdout.strip()