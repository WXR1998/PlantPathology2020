import pandas as pd
import os
import numpy as np

path = './result/EfficientNet'
files = ['result_1.csv', 'result_2.csv']

if __name__ == '__main__':

    result = None

    for f in files:
        full_path = os.path.join(path, f)
        df = pd.read_csv(full_path)
        if result is None:
            result = df
        else:
            for i in range(len(df)):
                result.iloc[i, 1:5] += df.iloc[i, 1:5]

    result.iloc[:, 1:5] /= len(files)
    result.to_csv(os.path.join(path, 'result.csv'), float_format='%.5f', index=False)