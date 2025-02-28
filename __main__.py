import os
import sys

from src.data.dataset import SpeakerDataset
from src.utils.config_manager import ConfigManager
conf = ConfigManager('configs/conf.json')

def test_dataset(data_path, speakers):
    data = SpeakerDataset(data_path, speakers)
    print(data.df.columns)
    print(data.df.size)
    print(f"features: {type(data.features)}, {data.features.shape}")
    print(f"labels: {data.labels}, {data.decode_label(0)}")

def test_svm():
    pass

def test_rf():
    pass

def test_ubm_gmm():
    pass

if __name__ == '__main__':
    database = conf.get('database.path')
    speakers = conf.get('database.speakers')
    test_dataset(database, speakers)

