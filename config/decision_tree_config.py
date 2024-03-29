import os

from easydict import EasyDict

from utils.enums import DataProcessTypes

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

cfg.wine_dataset_path = os.path.join(os.path.abspath(os.curdir), "../static", "wine-quality-white-and-red.csv")
