import ml_framework as mf 
import numpy as np
from helpers import make_data_from_csv_for_optionals, load_and_combine_csvs_for_optionals, confusion_matrix_multiclass, save_confusion_matrix_optionals, train_a_model_optionals




if __name__ == '__main__':
    # data generation
    # training data
    folder_path = "optional_gesture_data"
    X_train, labels_train = load_and_combine_csvs_for_optionals(folder_path)
    
    scaler = mf.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)


    y_train = mf.one_hot_encode(labels_train, 12)
    
    # VALIDATION data
    #FRAMES_FILE_PATH = "validation_data"
   
    #X_val, labels_val = load_and_combine_csvs(FRAMES_FILE_PATH)

    #X_val_scaled = scaler.transform(X_val)


  

    # compute class_weight for penalty
    sample_size = len(labels_train)
    unique_entries, counts = np.unique(labels_train, return_counts=True)
    weights = []
    for i in range(12):
        weights.append(1 / (counts[i] / sample_size))
    weights = np.array(weights)

    print(weights)



