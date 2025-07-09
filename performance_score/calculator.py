import argparse
import pandas as pd
# Example parameters:
#   --events_csv=demo_data/demo_video_events_rotate.csv
#   --ground_truth_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv

AVAILABLE_GESTURES = ["idle", "swipe_left", "swipe_right", "rotate"]


def read_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_csv",
                        help="CSV file containing a column 'events' with the events predicted by your model",
                        required=True)
    parser.add_argument("--ground_truth_csv",
                        help="CSV file containing a column 'ground_truth' with the correct gesture for each frame (may be the same as 'events_csv')",
                        required=True)

    args = parser.parse_known_args()[0]
    return args


def read_data(args):
    events = pd.read_csv(args.events_csv)
    ground_truth = pd.read_csv(args.ground_truth_csv)

    assert len(events) == len(ground_truth), "events and ground truth CSV are not of the same length!"

    # drop rows where no ground truth is given
    rows_to_drop = ground_truth["ground_truth"].isna().values
    ground_truth = ground_truth[~rows_to_drop]
    events = events[~rows_to_drop]

    return events, ground_truth


def count_individual_gestures(ground_truth):
    return (ground_truth.iloc[:-1] != ground_truth.shift(-1).iloc[:-1]).sum() // 2


def calculate_scores(events_df, ground_truth_df, bonus=10, malus=0.2):
    events = events_df["events"]
    ground_truth = ground_truth_df["ground_truth"]

    assert len(events) == len(ground_truth), "Error: the CSV files differ in length!"

    num_frames = len(ground_truth)
    num_total_gestures = count_individual_gestures(ground_truth)

    last_frame_gesture = "idle"
    current_gesture_detected = False

    marks = 0

    for frame_idx in range(num_frames):
        current_gesture = ground_truth.iloc[frame_idx]
        current_event = events.iloc[frame_idx]

        event_fired = current_event != "idle"
        gesture_in_progress = current_gesture != "idle"

        fired_wrong_event = gesture_in_progress and \
                            event_fired and \
                            current_event != current_gesture

        fired_event_but_no_gesture_in_progress = event_fired and not gesture_in_progress
        fired_event_more_than_once = event_fired and current_gesture_detected

        if last_frame_gesture != current_gesture:
            gesture_ended_but_no_event_fired = not gesture_in_progress and not current_gesture_detected
            current_gesture_detected = False
        else:
            gesture_ended_but_no_event_fired = False

        if fired_event_but_no_gesture_in_progress or \
                fired_event_more_than_once or \
                fired_wrong_event or \
                gesture_ended_but_no_event_fired:
            marks -= malus
        elif gesture_in_progress and current_event == current_gesture:
            marks += 1
            current_gesture_detected = True

        last_frame_gesture = current_gesture

    total_points = num_total_gestures
    score = max((marks / total_points) * 100 + bonus, 0)

    print("marks: %d/%d" % (marks, total_points))
    print("performance score: %.2f%%" % score)


if __name__ == "__main__":
    args = read_command_line_arguments()
    events, ground_truth = read_data(args)
    calculate_scores(events, ground_truth)
