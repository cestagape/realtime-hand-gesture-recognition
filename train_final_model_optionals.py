import ml_framework as mf 
import numpy as np
from helpers import make_data_from_csv_for_optionals, load_and_combine_csvs_for_optionals, confusion_matrix_multiclass, save_confusion_matrix_optionals, train_a_model_optionals


if __name__ == '__main__':
    np.random.seed(60)
    # data generation
    
    # training data
    folder_path = "optional_gesture_data"
    X_train, labels_train = load_and_combine_csvs_for_optionals(folder_path)
    
    scaler = mf.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler.save(filename="scaler_optionals.npz")
    
    y_train = mf.one_hot_encode(labels_train, 12)

    # VALIDATION data
    #FRAMES_FILE_PATH = "validation_data"
    #X_val, labels_val = load_and_combine_csvs(FRAMES_FILE_PATH)

    #X_val_scaled = scaler.transform(X_val)


    
    


    y_train = mf.one_hot_encode(labels_train, 12)

    #HYPERPARAMETERS:
    # compute class_weight for penalty
    sample_size = len(labels_train)
    unique_entries, counts = np.unique(labels_train, return_counts=True)
    weights = []
    for i in range(12):
        weights.append(1 / (counts[i] / sample_size))
    weights = np.array(weights)

    n = 15000
    l_rate = 0.2
    l2 = 0.0001
    b = 512

    #Training
    model = train_a_model_optionals(X_train_scaled, y_train, n, l_rate, l2, weights, b)
    model.save(filename="model_optionals.npz")

    y_pred_train = model.predict(X_train_scaled)
    
    f1_score_macro_train = mf.f1_score_multiclass(y_pred_train, labels_train, 'macro')
    
    print(f"\nTrain f1 Score macro = {f1_score_macro_train}")
    