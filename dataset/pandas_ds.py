import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# From https://blog.godatadriven.com/fairness-in-pytorch
class PandasDataSet(TensorDataset):

    '''
    Usage:
    train_data = PandasDataSet(features_train, label_train)
    test_data = PandasDataSet(features_test, label_test)
    '''

    def __init__(self, df_features, df_labels):
        tensors = (self._df_to_tensor(df_features), self._df_to_tensor(df_labels, dim=0))
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df, dim=1):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        x = torch.from_numpy(df.values).float()

        return F.normalize(x, p=2, dim=dim)


def load_pandas(df_features, df_labels, **data):
    
    norm_label = data.get('norm_label', False)

    tensors = PandasDataSet(df_features, df_labels)
    loader = DataLoader(tensors, **data)

    print('# training samples:', len(tensors))
    print('# batches:', len(loader))

    return loader
