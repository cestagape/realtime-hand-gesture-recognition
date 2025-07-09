import argparse
import numpy as np
import pandas as pd
# Example parameters:
#   --events_csv=demo_data/demo_video_events_rotate.csv
#   --ground_truth_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv


import matplotlib.pyplot as plt

categories = ['idle', 'rotate', 'swipe_left', 'swipe_right']


parser = argparse.ArgumentParser()
parser.add_argument("--events_csv",
                    help="CSV file containing a column 'events' with the events predicted by your model",
                    required=True)
parser.add_argument("--ground_truth_csv",
                    help="CSV file containing a column 'ground_truth' with the correct gesture for each frame (may be the same as 'events_csv')",
                    required=True)

args = parser.parse_known_args()[0]


events = pd.read_csv(args.events_csv)
ground_truth = pd.read_csv(args.ground_truth_csv)

prediction_oh = np.eye(len(categories))[pd.Categorical(events.events, categories=categories).codes]
ground_truth_oh = np.eye(len(categories))[pd.Categorical(ground_truth.ground_truth, categories=categories).codes]



px = 1/plt.rcParams['figure.dpi']


fig, ax = plt.subplots( figsize=(10, (prediction_oh.shape[0]+30) * px))


ground_truth_oh[prediction_oh>0] = 2

ax.imshow(ground_truth_oh[:, 1:], aspect="auto", interpolation="none")

ax.set_title("events")

ax.set_yticks(range(0, len(prediction_oh),100))
ax.set_xticks(range(3))
ax.set_xticklabels(categories[1:])
ax.xaxis.tick_top()

fig.tight_layout()
fig.savefig("events_visualization.png")
plt.show()

