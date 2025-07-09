import os
import argparse
import pandas as pd
import numpy as np
import random
import ml_framework as mf 
from helpers import make_data_from_csv
# Example parameters:
#   --input_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv
#   --output_csv_name=log_emitted_events_output_csv_name.csv


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

frames = pd.read_csv(input_path, index_col="timestamp")
frames.index = frames.index.astype(int)
# ================================= your application =============================
# you should import and call your own application here
x_data, y_data = make_data_from_csv(frames)

scaler = mf.StandardScaler.load()
X_train_scaled = scaler.transform(x_data)

my_model = mf.Model()

my_model.add(mf.Dense(160, 128, activation="sigmoid"))  
my_model.add(mf.Dense(128, 128, activation="sigmoid"))
my_model.add(mf.Dense(128, 4))                                       # may change

my_model.load()  
#print(np.argmax(my_model.predict(X_train_scaled), axis=1))
# ================================================================================

# determine events
frames["events"] = my_model.compute_events(np.argmax(my_model.predict(X_train_scaled), axis=1))

# the CSV has to have the columns "timestamp" and "events"
# but may also contain additional columns, which will be ignored during the score evaluation
frames["events"].to_csv(output_path, index=True) # since "timestamp" is the index, it will be saved also
print("events exported to %s" % output_path)