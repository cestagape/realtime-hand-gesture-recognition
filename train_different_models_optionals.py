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


    
    for n in [10000]:
        for l_rate in [0.001, 0.01, 0.1, 0.2]:
            for l2 in [0.01, 0.001, 0.0001]:
                for b in [256, 512]:
                    np.random.seed(60)
                    print(f"epochs={n}, learning rate = {l_rate}, l2 = {l2}, batchsize = {b}:", end='\n')
                    model = train_a_model_optionals(X_train_scaled, y_train, n, l_rate, l2, weights, b)

                    y_pred_train = model.predict(X_train_scaled)

                    acc_train = mf.accuracy(y_pred_train, y_train)

                    f1_score_macro_train = mf.f1_score_multiclass(y_pred_train, labels_train, 'macro')

                    confusion_train = confusion_matrix_multiclass(y_pred_train, labels_train)

                    save_confusion_matrix_optionals(confusion_train, n, l_rate, l2, f1_score_macro_train, acc_train, b)

