## Mandatory Requirements

### M4

Use:
```
pip install -r requirements.txt
```

### M3

Use
```
python slideshow/slideshow_demo.py
```
to run a slideshow that can be controlled via mandatory gestures.

Use
```
python slideshow/slideshow_demo_optionals.py
```
to run a slideshow that can be controlled via mandatory gestures and "flip_table" (shuffles slides"), "swipe_up"/"swipe_down", "spread" (zoom out) and "pinch" (zoom in).

### M6 Performance Score

`performance_score/log_emitted_events_to_csv.py` reads a CSV-file and produces a new CSV file (called `demo_data/emitted_events.csv`) containing a column events that contains the events your application registers for each frame.

Use:
```
python performance_score/log_emitted_events_to_csv.py --input_csv=demo_data/demo_video_frames_rotate.csv
```
assuming `demo_data/demo_video_frames_rotate.csv` is the given CSV.

`performance_score/calculator.py` can be used to determine the performance score.

Use:
```
python performance_score/calculator.py --events_csv=demo_data/emitted_events.csv --ground_truth_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv
```

`performance_score/events_visualization.py` can be used to visually compare the output of the model with the ground truth.

Use:
```
python performance_score/events_visualization.py --events_csv=demo_data/emitted_events.csv --ground_truth_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv
```


## Data Sets 

### `gesture_data/`: Train Set

The files are from the shared gesture dataset gitlab `https://gitlab2.informatik.uni-wuerzburg.de/s395069/gesture-dataset`.

### `validation_data/`: Validation Set

The files are provided by the supervisors.

### `optional_gesture_data/`: Train Set

The files are from the shared gesture dataset gitlab `https://gitlab2.informatik.uni-wuerzburg.de/s395069/gesture-dataset` for optional gestures.

## Training

`train_different_models.py` can be used to train different models with different hyperparameters. It produces confusion matrices that are stored in `con_matrices/`.

`train_final_model.py` was used to train the final model and to save it in `model.npz` and its feature scaler `scaler.npz`.

---

## Optional Requirements Implemented

### O2: Principal Component Analysis (PCA)
- PCA implemented in `project/pca.py`
- Applies PCA on gesture data after standard scaling
- Visualizes explained variance using matplotlib`


### O12: Gradient Descent Variations
- Implemented in `project/gradient_descent.py`
- Includes:
  - Vanilla Gradient Descent
  - Momentum-based Gradient Descent
  - Nesterov Accelerated Gradient (NAG)
- Simulates training and compares optimizers visually

