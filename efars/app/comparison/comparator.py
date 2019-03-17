"Comparator Lib, includes all stuff for comparing and analyzing results"
import datetime
import functools
import math
import multiprocessing
import os
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import dateutil.parser
import pandas
import numpy


class Comparator():
    def __init__(self, run_config):
        self.run_config = run_config
        self.pool = Pool(self.run_config.evaluation_pool_size)
        # clean the evaluation results folder
        if os.path.isdir(self.run_config.evaluation_plots_folder_path):
            shutil.rmtree(self.run_config.evaluation_plots_folder_path)
        pathlib.Path(self.run_config.evaluation_plots_folder_path).mkdir(
            parents=True, exist_ok=True)
        # get all runs
        self.runs = self.get_runs()
        # temporary config, for selcting only necesssary data
        self.run_groups = {'cpu_basic-recommender': 'CPU Basic Recommender', 'cpu_database': 'CPU Database',
                           'memory_basic-recommender': 'Memory Basic Recommender', 'memory_database': 'Memory Database',
                           'blkio_basic-recommender': 'Block IO Basic Recommender', 'blkio_database': 'Block IO Database'}

    def get_runs(self):
        return [f for f in listdir(self.run_config.runs_folder_path) if not isfile(join(self.run_config.runs_folder_path, f))]

    def precalc_provider_data(self):
        " Transform provider log data into meaningful data (aggregration)"
        temp_result = {}
        for run in self.runs:
            category_run = run[:run.rfind('_')]
            if not category_run in temp_result.keys():
                temp_result[category_run] = pandas.DataFrame()

            provisions_run_fp = os.path.join(
                self.run_config.runs_folder_path, run, "provisions", "provision_counts.csv")
            df = pandas.read_csv(provisions_run_fp, header=None, names=[
                'read_at', 'provisions_cnt', 'provisions_failed_cnt'])
            arr = numpy.arange(
                len(df)) // self.run_config.fetch_recommendations_skip_steps
            csum_df = df[['provisions_cnt', 'provisions_failed_cnt']].groupby(
                arr).cumsum().shift()
            last_tick = len(df) - 1
            # calculate amount of grouped ticks
            grouped_ticks = math.floor(
                last_tick / self.run_config.fetch_recommendations_skip_steps)

            # generate index to get every nth row
            every_nth_row_idx = range(
                0, last_tick, self.run_config.fetch_recommendations_skip_steps)
            every_nth_row_idx = list(every_nth_row_idx) + [last_tick]

            # merge cumsum and every nth row
            csum_df = csum_df.loc[every_nth_row_idx, :]
            df = df.loc[every_nth_row_idx, ['read_at']]
            df['provision_cnt'] = csum_df['provisions_cnt']
            df['provisions_failed_cnt'] = csum_df['provisions_failed_cnt']
            temp_result[category_run] = temp_result[category_run].append(
                df, sort=False)
        # clear folder
        csv_provider_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "provider")
        if os.path.isdir(csv_provider_folder):
            shutil.rmtree(csv_provider_folder)

        for cat_run, df in temp_result.items():
            avg = pandas.DataFrame()
            df['read_at'] = df['read_at'].map(
                lambda x: dateutil.parser.parse(x).timestamp())
            avg = df[['read_at', 'provision_cnt', 'provisions_failed_cnt']].groupby(
                df.index).mean()
            avg['read_at'] = avg['read_at'].map(
                lambda x: datetime.datetime.utcfromtimestamp(x))
            avg['time_elapsed'] = avg['read_at'] - avg['read_at'][0]
            avg['time_elapsed'] = avg['time_elapsed'].map(
                lambda x: x.total_seconds())
            pathlib.Path(csv_provider_folder).mkdir(
                parents=True, exist_ok=True)
            avg.to_csv(os.path.join(csv_provider_folder,
                                    "{cat_run}.csv".format(cat_run=cat_run)))

    def precalc_monitor_data(self):
        """ reads the hardware files and constructs an processable dict structure"""
        temp_result = {}
        for run in self.runs:
            category_run = run[:run.rfind('_')]
            if not category_run in temp_result.keys():
                temp_result[category_run] = {}

            run_folder_path = os.path.join(
                self.run_config.runs_folder_path, run)
            monitor_folder_path = os.path.join(run_folder_path, "monitor")
            results_folder_path = os.path.join(run_folder_path, "result")
            pathlib.Path(results_folder_path).mkdir(
                parents=True, exist_ok=True)
            hardware_info_files = [f for f in listdir(
                monitor_folder_path) if isfile(join(monitor_folder_path, f))]
            hardware_info_files.sort()
            for file in hardware_info_files:
                info_type = file[:-4]
                if not info_type in temp_result[category_run].keys():
                    temp_result[category_run][info_type] = pandas.DataFrame()
                df = pandas.read_csv(os.path.join(
                    monitor_folder_path, file))
                # index for all
                if info_type.startswith("blkio"):
                    df['bytes_read_sum'] = get_blkio_sum_devices(df, "Read")
                    df['bytes_write_sum'] = get_blkio_sum_devices(df, "Write")
                df['read_at'] = df['read_at'].map(
                    lambda x: dateutil.parser.parse(x))
                df['time_elapsed'] = df['read_at'] - df['read_at'][0]
                df['time_elapsed'] = df['time_elapsed'].map(
                   lambda x: x.total_seconds())
                temp_result[category_run][info_type] = temp_result[category_run][info_type].append(
                    df, sort=False)

        # clear folder
        csv_monitor_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "monitor")
        if os.path.isdir(csv_monitor_folder):
            shutil.rmtree(csv_monitor_folder)
        for run_type, monitored_data_types in temp_result.items():
            avgs = {}
            for monitored_data_type, df in monitored_data_types.items():
                avg_metrics = df.groupby(df.index, as_index=False).mean()
                group = "general"
                if monitored_data_type in self.run_groups:
                    group = self.run_groups[monitored_data_type]
                if monitored_data_type.startswith('cpu'):
                    # following https://github.com/docker/docker-ce/blob/222348eaf2226f0324a32744ad06d4a7bfe789ac/components/cli/cli/command/container/stats_helpers.go#L186
                    # but slightly different
                    cpu_delta = avg_metrics['cpu_stats_cpu_usage_total_usage'] - \
                        avg_metrics['cpu_stats_cpu_usage_total_usage'].shift(1)
                    system_delta = avg_metrics['cpu_stats_system_cpu_usage'] - \
                        avg_metrics['cpu_stats_system_cpu_usage'].shift(1)
                    avg_metrics['avg_cpu'] = cpu_delta / system_delta
                # write csv to temp folder:
                group_path = os.path.join(csv_monitor_folder, group)
                pathlib.Path(group_path).mkdir(parents=True, exist_ok=True)
                avg_metrics.to_csv(os.path.join(
                    group_path, "{run_type}.csv".format(group=group, run_type=run_type)))

    def load_monitor_csvs(self):
        final_result = {}
        csv_monitor_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "monitor")
        groups = [f for f in os.listdir(csv_monitor_folder) if not os.path.isfile(
            os.path.join(csv_monitor_folder, f))]
        for group in groups:
            group_fp = os.path.join(csv_monitor_folder, group)
            files = [f for f in os.listdir(
                group_fp) if os.path.isfile(os.path.join(group_fp, f))]
            if not group in final_result.keys():
                final_result[group] = {}
            for csv_file in files:
                run_type = csv_file[:-4]
                load_path = os.path.join(group_fp, csv_file)
                final_result[group][run_type] = pandas.read_csv(load_path)
        self.monitor_data = final_result

    def load_fetch_csvs(self):
        final_result = {}
        csv_fetch_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "fetch")
        fetches = [f for f in os.listdir(csv_fetch_folder) if os.path.isfile(
            os.path.join(csv_fetch_folder, f))]
        for csv_file in fetches:
            cat_run = csv_file[:-4]
            load_path = os.path.join(csv_fetch_folder, csv_file)
            final_result[cat_run] = pandas.read_csv(load_path)
        self.fetch_metrics = final_result

    def precalc_fetch_data(self):
        test_items = self.read_test_items()
        uti_splitted = split_into_relevant_and_irrelevant(
            test_items, self.run_config.test_relevant_items_rating_threshold)

        runs_to_process = []
        for run in self.runs:
            category_run = run[:run.rfind('_')]
            fetches_file_path = os.path.join(
                self.run_config.runs_folder_path, run, "fetches.csv")
            # skip if this file does not exist yet
            if os.path.isfile(fetches_file_path):
                runs_to_process.append(run)
        f = functools.partial(read_fetch_metrics_for_run,
                              uti_splitted=uti_splitted, basepath=self.run_config.runs_folder_path, n_ratings=self.run_config.fetches_rating_n, ratings_threshold=self.run_config.test_relevant_items_rating_threshold)

        metrics_df_tupels = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            metrics_df_tupels = executor.map(f, runs_to_process)

        result = {}
        for df_tupel in metrics_df_tupels:
            run = df_tupel[0]
            df = df_tupel[1]
            category_run = run[:run.rfind('_')]
            if not category_run in result.keys():
                result[category_run] = pandas.DataFrame()
            result[category_run] = result[category_run].append(df, sort=False)

        # clear folder
        csv_fetch_folder = os.path.join(
            self.run_config.evaluation_csv_folder_path, "fetch")
        if os.path.isdir(csv_fetch_folder):
            shutil.rmtree(csv_fetch_folder)

        avgs = {}
        for cat_run, df in result.items():
            avg_metrics = df[['f1 score', 'precision', 'recall', 'read_at', 'failure_indice']].groupby(
                ['measurement_tick'], as_index=True).mean()
            avg_metrics['read_at'] = avg_metrics['read_at'].map(
                lambda x: datetime.datetime.utcfromtimestamp(x))
            avgs[cat_run] = pandas.concat([avg_metrics], axis=1)
        for cat_run, df in avgs.items():
            df['time_elapsed'] = df['read_at'] - df['read_at'][0]
            df['time_elapsed'] = df['time_elapsed'].map(
                lambda x: x.total_seconds())
            # write csv to temp folder:

            pathlib.Path(csv_fetch_folder).mkdir(parents=True, exist_ok=True)
            df.to_csv(os.path.join(csv_fetch_folder,
                                   "{cat_run}.csv".format(cat_run=cat_run)))

    def read_test_items(self):
        ratings_file = open(
            self.run_config.evaluation_test_dataset_filepath, 'r')
        items = {}
        # read all ratings:
        for line in ratings_file:
            rowData = line.rstrip(self.run_config.evaluation_source_file_newline).split(
                self.run_config.evaluation_source_file_delimiter)
            rowUserId = rowData[0]
            rowItemId = rowData[1].strip()
            rowItemRating = rowData[2].strip()
            if rowUserId in items:
                items[rowUserId].append((rowItemId, rowItemRating))
            else:
                items[rowUserId] = [(rowItemId, rowItemRating)]
        ratings_file.close()
        return items


def __calculate(arg, uti_splitted, ratings_threshold, n_rating):
    user_id, df = arg
    results = []
    rti = uti_splitted[user_id]['relevant']
    iti = uti_splitted[user_id]['irrelevant']

    for index, df_row in df.iterrows():
        recommended_items = df_row[df.drop(
            ['measurement_tick'], axis=1).columns[-2*n_rating:]]
        chunked_items = list(chunks(list(recommended_items), 2))
        rii = list(strip_ratings(filter_interesting_items(
            chunked_items, ratings_threshold)))
        rui = list(strip_ratings(filter_uninteresting_items(
            chunked_items, ratings_threshold)))
        ri = strip_ratings(chunked_items)
        ri = [x for x in ri if not math.isnan(float(x))]
        tp = [item for item in ri if item in rti]
        fp = [item for item in ri if item in iti]
        fn = list(set(rti) - set(rii))
        r = recall(tp, fn)
        p = precision(tp, fp)
        try:
            f1 = (2 * r * p) / (r + p)
        except ZeroDivisionError:
            f1 = 0

        # provide metrics data:
        df_row['recall'] = r
        df_row['precision'] = p
        df_row['f1 score'] = f1

        results.append(df_row)
    return pandas.DataFrame(results)


def read_fetch_metrics_for_run(run_folder, basepath, uti_splitted, n_ratings, ratings_threshold):
    fetches_file_path = os.path.join(basepath, run_folder, "fetches.csv")
    # calculate relevant and irrelevant items from test dataset for each user
    df = pandas.read_csv(fetches_file_path, dtype=str)

    # parallel processing
    df = df.sort_values(by=['user_id', 'measurement_tick'])
    grp_lst_args = df.groupby('user_id')
    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
    f = functools.partial(__calculate, uti_splitted=uti_splitted,
                          ratings_threshold=ratings_threshold, n_rating=n_ratings)
    results = pool.map(f, grp_lst_args)
    pool.close()
    pool.join()
    df = pandas.concat(results)
    df['measurement_tick'] = df['measurement_tick'].astype('int64')
    df = df.sort_values(by=['measurement_tick'])
    df['read_at'] = df['read_at'].map(
        lambda x: dateutil.parser.parse(x).timestamp())
    df['failure_indice'] = df['failure_indice'].astype('int64')
    df['user_id'] = df['user_id'].astype('int64')
    avg_metrics = df[['f1 score', 'precision',
                      'recall', 'read_at', 'measurement_tick']].groupby('measurement_tick').mean()
    avg_error = df[['failure_indice', 'measurement_tick']
                   ].groupby('measurement_tick').sum()
    df = pandas.concat([avg_metrics, avg_error], axis=1)
    return (run_folder, df)


def strip_ratings(items):
    return list(map(lambda x: x[0], items))


def filter_interesting_items(items, threshold):
    return filter(lambda x: (True if float(x[1]) >= threshold else False), items)


def filter_uninteresting_items(items, threshold):
    return filter(lambda x: (True if float(x[1]) < threshold else False), items)


def split_into_relevant_and_irrelevant(users_with_items, threshold):
    result = {}
    for user, items in users_with_items.items():
        relevant_items = list(
            strip_ratings(filter_interesting_items(items, threshold)))
        irrelevant_items = list(
            strip_ratings(filter_uninteresting_items(items, threshold)))
        result[user] = {"relevant": relevant_items,
                        "irrelevant": irrelevant_items}
    return result


def precision(tp, fp):
    if len(set(tp)) == 0:
        return 0
    return len(set(tp)) / len(set(tp + fp))


def recall(tp, fn):
    if len(set(tp)) == 0:
        return 0
    return len(set(tp)) / len(set(tp + fn))


def chunks(l, n):
    """ Create a function called "chunks" with two arguments, l and n: """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def get_blkio_sum_devices(df, op):
    op_cols = list(filter(lambda x: x.startswith(
        "io_service_bytes") and x.endswith("op"), df.columns))
    device_read_ops = list(
        filter(lambda x: df.iloc[0][x] == op, op_cols))
    device_ids = list(map(lambda x: x.lstrip(
        "io_service_bytes_recursive_").rstrip("_op"), device_read_ops))
    value_cols = list(map(lambda x: "io_service_bytes_recursive_{0}_value".format(x), device_ids))
    return df[value_cols].sum(axis=1)