import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

def load_dataset(file_name,header_name_set, header_todrop_set,
                 timestamp_header, number_lines):
    ds = pd.read_csv(file_name,
                       sep=",", header=None,
                       names=header_name_set)
    ds = ds.head(number_lines)
    print(ds.shape)
    ### drop columns
    for val in header_todrop_set:
        ds.drop(columns=[val], inplace=True)

    ds[timestamp_header] = [datetime.strptime(date, '%m/%d/%Y %H:%M') for date in ds[timestamp_header].values]



    return ds

def prepare_dataset (dataset, test_size, is_shuffle, target_field):
    dataset = dataset.loc[:, [target_field]]

    train_set, test_set = train_test_split(dataset, test_size=test_size, shuffle=is_shuffle)
    test_set = test_set.sort_index()
    train_set = train_set.sort_index()
    plt.title('test_set')
    plt.plot(test_set.index, test_set.values, 'g')
    plt.show()
    plt.title('train_set')
    plt.plot(train_set.index, train_set.values, 'g')
    plt.show()
    return train_set, test_set




''''
def convert_timestamp_ds(ds, timestamp_field):
    ## Convert the ts field (timestamp) to the correct format (datetime64)
    ds.index = pd.to_datetime((ds.index * 1e9).astype('int64'), utc=True)
    ds = ds.sort_index()

    return ds
'''





