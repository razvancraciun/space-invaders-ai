from matplotlib import pyplot as plt
import os

DIR = 'logs/'

logs = []

for filename in os.listdir(os.fsencode(DIR)):
    logs.append(os.fsdecode(filename))

logs = sorted(logs, key=lambda filename: int(filename[3:-4]) )

vals = []

for file in logs:
    f = open(DIR+file, 'r')
    values = f.read()[1:].split()
    for value in values:
        vals.append(int(value[:-3]))

plt.plot(vals)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
