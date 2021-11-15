import json
import os
from pprint import pprint
from matplotlib import pyplot as plt

FILEPATH = 'det_output.json'
FILEPATH = os.path.abspath(FILEPATH)

try:
    with open(FILEPATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print('File {} does not exist'.format(FILEPATH))
except Exception as e:
    print('Couldn\'t open file {}, {}'.format(FILEPATH, e))

fringe_counts = []

for imname, ostruct in data.items():
    dets = ostruct['det']
    for det in dets:
        try:
            if det['fringe_count'] is not None:
                    fringe_counts.append(det['fringe_count'])
        except Exception as e:
            print('whassaap')
            continue

print('we have {} fringe counts'.format(len(fringe_counts)))
plt.figure()
plt.hist(fringe_counts)
plt.show()

