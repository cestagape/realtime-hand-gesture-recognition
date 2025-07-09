import ml_framework as mf 
import numpy as np
from helpers import make_data_from_csv, load_and_combine_csvs, confusion_matrix_multiclass, save_confusion_matrix, train_a_model








if __name__ == '__main__':
    np.random.seed(60)
    # data generation
    
    # training data
    folder_path = "gesture_data"
    X_train, labels_train = load_and_combine_csvs(folder_path)
    
    scaler = mf.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler.save()
    
    y_train = mf.one_hot_encode(labels_train, 4)

    # VALIDATION data
    FRAMES_FILE_PATH = "validation_data"
    X_val, labels_val = load_and_combine_csvs(FRAMES_FILE_PATH)

    X_val_scaled = scaler.transform(X_val)



    #HYPERPARAMETERS:
    # compute class_weight for penalty
    sample_size = len(labels_train)
    unique_entries, counts = np.unique(labels_train, return_counts=True)
    weights = []
    for i in range(4):
        weights.append(1 / (counts[i] / sample_size))
    weights = np.array(weights)
    weights = weights / weights[0]
    weights = weights / 5
    weights[0] = 1

    n = 25000
    l_rate = 0.1
    l2 = 0.001
    b = 512

    #Training
    model = train_a_model(X_train_scaled, y_train, n, l_rate, l2, weights, b)
    model.save()

    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    f1_score_macro_train = mf.f1_score_multiclass(y_pred_train, labels_train, 'macro')
    f1_score_macro_validation = mf.f1_score_multiclass(y_pred_val, labels_val, 'macro')
    print(f"\nTrain f1 Score macro = {f1_score_macro_train}")
    print(f"\nValidation f1 Score macro = {f1_score_macro_validation}")