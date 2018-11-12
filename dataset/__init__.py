import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchtext import *

# From https://blog.godatadriven.com/fairness-in-pytorch
class PandasDataSet(TensorDataset):

    '''
    Usage:
    train_data = PandasDataSet(features_train, label_train)
    test_data = PandasDataSet(features_test, label_test)
    '''

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return torch.from_numpy(df.values).float()


def load_pandas(df_features, df_label, **data):

    tensors = PandasDataSet(df_features, df_label)
    loader = DataLoader(tensors, **data)

    print('# training samples:', len(tensors))
    print('# batches:', len(loader))

    return loader
