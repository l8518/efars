import datetime
import os
import pathlib
import shutil
import time
from functools import partial
from multiprocessing import Pool
import concurrent
import math
import pandas


class Provider():
    def __init__(self, emitter, source_file_path, provisions_per_tick, provider_pool_size, provisions_target_folder):
        self.emitter = emitter
        self.source_file_path = source_file_path
        self.source_file_delimiter = ","
        self.source_file_newline = "\r\n"
        self.ratings_file = open(self.source_file_path, 'r')
        self.provisions = 0
        self.provisions_per_tick = provisions_per_tick
        self.provider_pool_size = provider_pool_size
        self.provisions_target_folder = provisions_target_folder
        self.clean_up()
        self.pool_chunksize = max(1, math.floor( self.provisions_per_tick / provider_pool_size ) )
        self.worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.provider_pool_size)

    def tear_down(self):
        self.ratings_file.close()
        self.worker_pool.shutdown()

    def clean_up(self):
        """ cleans the run folder for run preparations """
        if os.path.isdir(self.provisions_target_folder):
            shutil.rmtree(self.provisions_target_folder)
        pathlib.Path(self.provisions_target_folder).mkdir(
            parents=True, exist_ok=True)

    def tick(self, tick):
        provision_data = []
        # read provisions from the data file
        for tick in range(self.provisions_per_tick):
            line = self.ratings_file.readline()
            data = line.rstrip(self.source_file_newline).split(
                self.source_file_delimiter)
            if line != "":
                user_id = data[0]
                item_id = data[1]
                rating_id = float(data[2])
                tupel = [user_id, item_id, rating_id]
                provision_data.append(tupel)
        # emit the provisions by using a pool
        f =  partial(emit, emitter=self.emitter)
        emitted_provisions = list(self.worker_pool.map(f, provision_data,chunksize=self.pool_chunksize))
        
        # log the information about provisions
        provisions_count = len(emitted_provisions) 
        error_count =  len([x for x in emitted_provisions if x[0] != 0])
        eventTime = str(datetime.datetime.utcnow().replace(
                tzinfo=datetime.timezone.utc).isoformat())
        filePath = os.path.join(
            self.provisions_target_folder, "provision_counts.csv")
        tick_result = {"read_at": [eventTime], "total_provisions": [provisions_count], "errors": [error_count]}
        df = pandas.DataFrame(tick_result)
        df.to_csv(filePath, mode='a', header=False, index=False)

def emit(provision, emitter):
    return emitter.emit_rating(provision[0], provision[1], provision[2])
