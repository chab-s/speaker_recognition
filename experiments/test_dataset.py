import os
import sys

from src.data.dataset import SpeakerDataset
from src.utils.config_manager import ConfigManager
conf = ConfigManager('configs')

def build_dataset(data_path, speakers):
    data = SpeakerDataset(data_path, speakers)
    print(data.df.columns)
    print(data.df.size)
    print(f"features: {type(data.features)}, {data.features.shape}")
    print(f"labels: {data.labels[:5]}, {data.decode_label(data.labels)[0]}")

if __name__ == '__main__':
    database = conf.get('database.path')
    speakers = conf.get('database.speakers')
    build_dataset(database, speakers)