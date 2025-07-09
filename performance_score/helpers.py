import ml_framework as mf 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

STEPSIZE = 10
Y_THRESHOLD = 100

def make_data_from_csv(csv, stepsize=STEPSIZE, y_threshold=Y_THRESHOLD):
    """
    Convert a csv file like demo_video_csv_with_ground_truth_rotate.csv into x and y data for training
    
    :param csv: csv file (should look like demo_video_csv_with_ground_truth_rotate).
    :param stepsize: number of frames per sample (= size of one window).
    :param y_threshold: one sample is labeled as a gesture if over y_threshold % of the frames belongs to the gesture.
    :return: X_array, y_array.
    """
    f = csv[["left_elbow_x", "left_elbow_y", "left_elbow_z", 
                "left_elbow_confidence", "left_wrist_x", "left_wrist_y", 
                "left_wrist_z", "left_wrist_confidence", "right_elbow_x", 
                "right_elbow_y", "right_elbow_z", "right_elbow_confidence", 
                "right_wrist_x", "right_wrist_y", "right_wrist_z", "right_wrist_confidence"]]
    # Define the step size in rows
    step = stepsize  
    
    # Get the total number of rows
    num_rows = len(f)
    
    # Generate the array with flattened 60-row chunks
    X_array = np.array([
        f.iloc[i : i + step].to_numpy().flatten()
        for i in range(0, num_rows-step, 1)
    ])
    
    # y DATA:
    if "ground_truth" in csv.columns:
        g = csv[["ground_truth"]]
        y_array = np.array([
            g.iloc[i : i + step].to_numpy().flatten()
            for i in range(0, num_rows-step, 1)
        ])
        
        klasses = []
        for i, window in enumerate(y_array):
            unique_entries, counts = np.unique(window, return_counts=True)
            rotate_percentage = counts[unique_entries == "rotate"] / len(window) * 100
            swipe_right_percentage = counts[unique_entries == "swipe_right"] / len(window) * 100
            swipe_left_percentage = counts[unique_entries == "swipe_left"] / len(window) * 100
            klass = 0
            if rotate_percentage.size > 0 and rotate_percentage >= y_threshold:
                klass = 1
            elif swipe_right_percentage.size > 0 and swipe_right_percentage >= y_threshold:
                klass = 2
            elif swipe_left_percentage.size > 0 and swipe_left_percentage >= y_threshold:
                klass = 3
            klasses.append(klass)
        y_array = np.array(klasses)
    else:
        y_array = None
    return X_array, y_array


def load_and_combine_csvs(folder_path, stepsize=STEPSIZE, y_threshold=Y_THRESHOLD):
    all_X = []
    all_y = []

    # Iterate over all CSV files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            csv_data = pd.read_csv(file_path)

            # Process each file
            X_array, y_array = make_data_from_csv(csv_data, stepsize, y_threshold)

            # Append to lists
            all_X.append(X_array)
            all_y.append(y_array)

    # Concatenate all arrays
    if all_X:
        X_final = np.vstack(all_X)
        y_final = np.hstack(all_y)
        return X_final, y_final
    else:
        return None, None  # Return None if no valid data found





def confusion_matrix_multiclass(h, y):
    """
    Compute the confusion matrix for multi-class classification.
    
    :param h: Array of predicted labels.
    :param y: Array of true labels.
    :return: Confusion matrix (2D NumPy array).
    """
    classes = np.unique(y)  # Get unique class labels
    
    num_classes = len(classes)
    
    # Initialize confusion matrix with zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    h = np.argmax(h, axis=1)  # Convert probabilities to predicted labels

    # Fill the confusion matrix
    for true_label, pred_label in zip(y, h):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            conf_matrix[true_label, pred_label] += 1
        else:
            print(f"Warning: Ignoring out-of-bounds label ({true_label}, {pred_label} , {num_classes})")
    
    return conf_matrix

def save_confusion_matrix(conf_matrix, n, l_rate, l2, f1train, f1val, acctrain, accval, batchsize, class_names=["idle", "rotation", "swipe right", "swipe left"]):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"con_matrices/itr={n}_lr={l_rate}_l2={l2}_f1train={f1train:.2%}_f1val={f1val:.2%}_acctrain={acctrain:.2%}_accval={accval:.2%}_batchsize={batchsize}_confusion_matrix.png")
    plt.close()

def train_a_model(X, y, epochs, lr, l2_lambda, weights, b):
    # Create a model
    model = mf.Model()
    model.add(mf.Dense(240, 256, activation="sigmoid", l2_lambda=l2_lambda))  
    model.add(mf.Dense(256, 128, activation="sigmoid", l2_lambda=l2_lambda))
    model.add(mf.Dense(128, 4, l2_lambda=l2_lambda))  # Output layer (No activation)

    # Compile with loss and optimizer
    model.compile(loss=mf.CrossEntropy(class_weights={0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3]}), optimizer=mf.GradientDescent(learning_rate=lr))

    # Train model
    model.train(X, y, epochs=epochs, batch_size=b)
    return model