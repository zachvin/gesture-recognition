import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

DARKBLUE = '#354A5A'
LIGHTBLUE = '#6589A4'
RED = '#DB5461'

parser = argparse.ArgumentParser()

parser.add_argument('--file', help='File path')

args = parser.parse_args()

files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
most_recent_file = max(files, key=os.path.getmtime)

filepath = args.file if args.file is not None  else most_recent_file
print(f'Plotting file {filepath}')

with open(filepath, 'r') as f:
    training_loss = []
    training_accuracy = []
    testing_accuracy = []
    for row in f:
        if 'Saving' in row:
            break
        if 'Training loss' in row:
            training_loss.append(float(row.split(' ')[-1].strip()))
        if 'Training accuracy' in row:
            training_accuracy.append(float(row.split(' ')[-1].strip()[:-2]))
        if 'Testing accuracy' in row:
            testing_accuracy.append(float(row.split(' ')[-1].strip()[:-2]))

x = range(1, len(training_loss)+1)

fig, ax1 = plt.subplots()

plt.title('Training data')
plt.axhline(y=sum(testing_accuracy[-10:])/len(testing_accuracy[-10:]), color=DARKBLUE, alpha=0.7, linestyle='dashed')

ax1.plot(x, testing_accuracy, color=DARKBLUE, label='Testing accuracy')
ax1.plot(x, training_accuracy, color=LIGHTBLUE, label='Training accuracy')
ax1.set_ylim([0, 100])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.tick_params(axis='y', labelcolor=DARKBLUE)

ax2 = ax1.twinx()
ax2.plot(x, training_loss, color=RED, alpha=0.5, label='Training loss')
ax2.set_ylabel('Training loss')
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim([0,8])

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('train_fig.png')
