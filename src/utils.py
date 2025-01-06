import numpy as np
import csv
from time import time
import numpy
from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMinMax

def extract_ts_from_file(file_path):

    line_ts = []
    with open(file_path) as file:
        tsv_file = csv.reader(file, delimiter = '\t')
        for line in tsv_file :
            line_ts.append(line)

    line_ts = np.array(line_ts)
    
    classes = np.array(line_ts[:, 0], dtype = int)
    ts_array = np.array(line_ts[:, 1:], dtype = np.float32)

    return classes, ts_array

def get_statistics_models(models, train_set_path, test_set_path):
    # Load the data
    train_labels, train_ts = extract_ts_from_file(train_set_path)
    test_labels, test_ts = extract_ts_from_file(test_set_path)

    # # Normalize
    # scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))
    # train_ts_scaled = scaler.fit_transform(train_ts)
    # test_ts_scaled = scaler.fit_transform(test_ts)

    # # Shuffle the data
    # indices_shuffle = numpy.random.permutation(len(train_labels))
    # y_train_shuffle = train_labels[indices_shuffle]
    # X_train_shuffle = train_ts_scaled[indices_shuffle,:]

    # y_test_shuffle = test_labels[indices_shuffle]
    # X_test_shuffle = test_ts_scaled[indices_shuffle,:]


    accuracy, time_complexity = dict(), dict()

    for model_name, model in models.items():
            start = time()
            # Train the model
            model.fit(train_ts, train_labels)
            end_training = time()
            # Predict the labels for the test set 
            prediction = model.predict(test_ts)
            end_prediction = time()
            # Calculate the accuracy
            accuracy[model_name] = accuracy_score(prediction, test_labels)
            # Calculate the time complexity
            time_complexity[model_name] = end_prediction - start
            print(f"{model_name} accuracy: {accuracy[model_name]}, time complexity: {time_complexity[model_name]} seconds")
    return accuracy, time_complexity

def read_ts(path): 
    time_series_list = []
    labels = []

    # Read the file and process each line
    with open(path, "r") as file:
        for line in file:
            # Split the line into time series and label
            time_series_str, label_str = line.strip().split(":")
            # Convert the time series to a numpy array of floats
            time_series = np.array([float(x) for x in time_series_str.split(",")])
            
            # Append to the lists
            time_series_list.append(time_series)
            labels.append(int(label_str))  # Convert label to int (or float if needed)

    # Convert lists to numpy arrays
    time_series_array = np.array(time_series_list)  # 2D array where each row is a time series
    labels_array = np.array(labels)  # 1D array of labels
    return time_series_array, labels_array

def get_statistics_models_from_data(models, train_ts, train_labels, test_ts,test_labels):
    accuracy, time_complexity = dict(), dict()

    for model_name, model in models.items():
            start = time()
            # Train the model
            model.fit(train_ts, train_labels)
            end_training = time()
            # Predict the labels for the test set 
            prediction = model.predict(test_ts)
            end_prediction = time()
            # Calculate the accuracy
            accuracy[model_name] = accuracy_score(prediction, test_labels)
            # Calculate the time complexity
            time_complexity[model_name] = end_prediction - start
            print(f"{model_name} accuracy: {accuracy[model_name]}, time complexity: {time_complexity[model_name]} seconds")
    return accuracy, time_complexity
