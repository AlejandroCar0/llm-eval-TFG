import os
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
def read_prompts() -> list:
    prompts = []
    with open(f"{EXECUTION_PATH}/prompts.txt","r") as promptsList:
        for line in promptsList:
            line = line.rsplit("\;")
            prompts.append(line)

    return prompts

read_prompts()