import numpy as np
import csv


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