from easydict import EasyDict
import pandas as pd

from datasets.base_dataset_classes import BaseDataset
from utils.common_functions import read_dataframe_file


class WineDataset(BaseDataset):

    def __init__(self, cfg: EasyDict):
        super().__init__(cfg.train_set_percent, cfg.valid_set_percent)

        wine = read_dataframe_file(cfg.wine_dataset_path)

        # encode type
        # we need to use label encoding, but we can just replace
        # red with 1 and white with 0 (for example)
        # because we can drop first column
        wine_red_mask = wine['type'] == 'red'
        wine_type_encoded = wine_red_mask.astype(int)
        wine['type'] = wine_type_encoded

        self._targets = wine['quality'].to_numpy()
        self._inputs = wine.drop(columns=['quality']).to_numpy()
        self._d = len(self._inputs[0])

        self.divide_into_sets()

    @property
    def targets(self):
        return self._targets

    @property
    def inputs(self):
        return self._inputs

    @property
    def d(self):
        return self._d
