import json
from colorama import Fore

instructions=[]
with open('model1data.json','r') as f:
    data=json.load(f)
    for key,chunk in data.items():
        for pairs in chunk['generated']:
            instructions.append(pairs)
        print(Fore.YELLOW+str(chunk))
        print('\n____________')

with open('instructions.json','w') as f:
    json.dump(instructions,f,indent=4)


with open('instructions.json','r') as f:
        data = json.load(f)
        print(Fore.LIGHTMAGENTA_EX + str(data))