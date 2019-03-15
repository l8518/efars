import requests
import zipfile
import io
import random
from abc import ABC, abstractmethod
import os


class Planner():

    def __init__(self):
        pass

    def download(self, url):
        request = requests.get(url)
        file = zipfile.ZipFile(io.BytesIO(request.content))
        file.extractall("./data/")
        return 

class DatasetAdapter(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, source_file_path, target_folder, seed, split_ratio, rating_threshold, user_cnt ):
        raise NotImplementedError