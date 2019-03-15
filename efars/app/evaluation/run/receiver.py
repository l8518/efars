from multiprocessing import Pool
from functools import partial
import requests
import datetime
import os
import json
import shutil
import time
import pathlib
import pandas
import math
import concurrent
import threading


class Receiver():
    def __init__(self, source_file_path, target_folder, receiver_pool_size, receiver_adapter_instance, rating_n):
        """ Initializes a new recevier. \n

        The receiver fetches data from the system under test, by using the passed fetcher
        Call in a loop to create terminal progress bar
        @params:
            TODO   - Required  : TODO (Int)
            TODO   - Required  : TODO (Int)
            TODO   - Optional  : TODO (Str)
        """
        self.source_file_path = source_file_path
        self.source_file_delimiter = ","
        self.source_file_newline = "\r\n"
        self.users = self.get_users_with_items()
        self.target_folder = target_folder
        self.clean_up()
        self.receiver_pool_size = receiver_pool_size
        self.rating_n = rating_n
        self.receiver_adapter_instance = receiver_adapter_instance
        # keep worker pool open, saves spwaning time
        self.pool_chunksize=max(1, math.floor( len(self.users) / self.receiver_pool_size ) )
        self.worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.receiver_pool_size)

    def get_users_with_items(self):
        """ gets the user ids for fetching """
        # TODO: This is data file dependent!
        # TODO: this is an old implementation, that should only return the ids of users
        # TODO: -> strip old functiontionality
        ratings_file = open(self.source_file_path, 'r')
        users_with_items = {}
        # read all ratings:
        for line in ratings_file:
            data = line.rstrip(self.source_file_newline).split(
                self.source_file_delimiter)
            userId = data[0]
            movieId = data[1]
            ratingId = data[2]
            if userId not in users_with_items.keys():
                users_with_items[userId] = [(movieId, float(ratingId))]
            else:
                users_with_items[userId].append((movieId, float(ratingId)))
        ratings_file.close()
        return users_with_items.keys()

    def clean_up(self):
        """ cleans the run folder for run preparations """
        if os.path.isdir(self.target_folder):
            shutil.rmtree(self.target_folder)
        pathlib.Path(self.target_folder).mkdir(parents=True, exist_ok=True)

    def tear_down(self):
        self.worker_pool.shutdown()

    def fetch(self, tick):
        """ fetches recommendations for a set of users """
        fetched_users = []
        f = partial(fetch_and_write_user, rating_n=self.rating_n, receiver_adapter_instance=self.receiver_adapter_instance)
        fetched_users = list( self.worker_pool.map(f, self.users , chunksize=self.pool_chunksize) )
        # Write File in Separate File
        filePath = os.path.join(self.target_folder, "{0}.csv".format(tick)) 
        WriteFetches(fetched_users, filePath).start() # no wait for this required

    def join_fetches(self):
        """ Aggregates all fetches that were obtained during a run and writes them to a csv file in the run folder """
        #
        aggregated_fetches_filep = os.path.join(
            os.path.dirname(self.target_folder),  "fetches.csv")
        users_files = [f for f in os.listdir(
            self.target_folder) if os.path.isfile(os.path.join(self.target_folder, f))]
        users_files.sort()
        aggregated_fetches_dfs = []

        rating_n = range(self.rating_n)
        for user_fetches_filename in users_files:
            measurement_tick = user_fetches_filename[:-4]
            fetches_fp = os.path.join(
                self.target_folder, user_fetches_filename)
            rating_names = [i for j in rating_n for i in [
                "item_{}".format(j), "itemrating_{}".format(j)]]
            df = pandas.read_csv(fetches_fp, header=None, names=[
                'read_at', 'user_id', 'failure_indice'] + rating_names, dtype=str)
            #df['failure_indice'] = df['failure_indice'].astype('int64')
            df['measurement_tick'] = int(measurement_tick)
            
            aggregated_fetches_dfs.append(df)
        aggreagated_df = pandas.concat(aggregated_fetches_dfs)
        aggreagated_df = aggreagated_df.sort_values(by=['user_id', 'measurement_tick'])
        aggreagated_df.to_csv(aggregated_fetches_filep,  index=False)

        """ fetches recommendations for a set of users """

def fetch_and_write_user(user, rating_n, receiver_adapter_instance):
    """ fetches recommendation for a user and writes the recommendation with meta data to a csv file """

    # fetch the recommendations
    item_scores = receiver_adapter_instance.fetch_user(user, rating_n)

    # take time of the event
    eventTime = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # Construct the default line
    line = [eventTime, user, '1'] + ['', ''] * rating_n
    if item_scores is not None:
        line = [eventTime, user, '0'] + item_scores
    return line

class WriteFetches(threading.Thread):  
  
    def __init__(self, fetched_users, fp): 
        threading.Thread.__init__(self)
        self.fetched_users = fetched_users 
        self.fp = fp 
  
    def run(self): 
        df = pandas.DataFrame(self.fetched_users)
        df.to_csv(self.fp, mode='a', header=False)