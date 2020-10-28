import json
import os

with open("../data/train.json", "r") as f:
    data = json.load(f)
    for i in range(int(len(data))):
        code = data[i]["code"]
        code = ' '.join(code)
        with open("../data/train.txt", "a") as f:
            f.write("\n")
            f.write(code)

