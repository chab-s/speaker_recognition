import kagglehub
import os

class DataManager:
    def __init__(self, dataset: str, download_path: str = None):
        self.dataset = dataset
        self.download_path = download_path if download_path else os.path.expanduser(
            f"~/.cache/kagglehub/{dataset}")
        os.makedirs(self.download_path, exist_ok=True)
        print(self.download_path)

    def import_dataset(self):
        if os.path.exists(self.download_path):
            print(f"Dataset '{self.dataset}' already downloaded at {self.download_path}")
            return self.download_path

        print(f"Downloading dataset '{self.dataset}'...")
        path = kagglehub.dataset_download(self.dataset, path=self.download_path)
        return path

