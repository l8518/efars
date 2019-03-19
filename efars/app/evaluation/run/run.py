import math
import os
import subprocess
import time
from shutil import copyfile

from .monitor import Monitor
from .provider import Provider
from .receiver import Receiver


class Run():

    def __init__(self, run_config):
        self.run_config = run_config

    def start(self):
        """ starts the runs of a configuration """
        self.run_times = {}
        for run_index in range(self.run_config.run_n):
            # construct relevant information / parameters
            runname = "{0}_{1}".format(self.run_config.run_prefix, run_index)

            # run
            print("####### Run {0} from {1}".format(run_index + 1, self.run_config.run_n))
            subprocess.call("docker-compose up -d", shell=True)
            time.sleep(self.run_config.run_warm_up_seconds)
            time_start_and_end = self.single_run(run_index, runname)
            subprocess.call("docker-compose down", shell=True)

            # document relevant metrics:
            self.run_times[runname] = time_start_and_end
        self.document_runs()
        self.persist_test_files()

    def persist_test_files(self):
        folder_pt = os.path.join(
            self.run_config.data_root_folder,
            "test_files",
            self.run_config.run_prefix
        )
        pathlib.Path(folder_pt).mkdir(parents=True, exist_ok=True)
        # define file targets
        test_fp = os.path.join(folder_pt, "test.csv")
        test_relevant_fp = os.path.join(folder_pt, "test_irrelevant.csv")
        test_irrelevant_fp = os.path.join(folder_pt, "test_relevant.csv")
        copyfile(self.run_config.test_source_file, test_fp)
        copyfile(self.run_config.test_relevant_source_file, test_relevant_fp)
        copyfile(
            self.run_config.test_irrelevant_source_file,
            test_irrelevant_fp)

    def document_runs(self):
        """ Documents the run parameters for this run"""
        fw = open(os.path.join(self.run_config.data_root_folder,
                               "{0}_logfile".format(self.run_config.run_prefix)), "w")
        fw.write(self.run_config.get_printable_config())
        for runname, run_times in self.run_times.items():
            fw.write("Overall Duration (in s) for run {0}:\t {1}  \n".format(runname,
                                                                               run_times[1] - run_times[0]))
        fw.flush()
        fw.close()

    def single_run(self, run_index, runname):
        """ Executes a single run for the given configuration """
        printProgressBar(
            0, self.run_config.max_ticks, prefix='Progress:', suffix='Complete', length=50)

        fetches_target_folder = os.path.join(
            "data", "runs", "{0}".format(runname), "fetches")

        provisions_target_folder = os.path.join(
            "data", "runs", "{0}".format(runname), "provisions")

        provider = Provider(
            self.run_config.provider_adapter_instance, self.run_config.training_source_file, self.run_config.provisions_per_tick, self.run_config.concurrent_provisions, provisions_target_folder)
        monitor = Monitor(self.run_config.docker_containers, self.run_config.docker_sock, runname)

        receiver = Receiver(
            self.run_config.test_source_file,
            fetches_target_folder,
            self.run_config.concurrent_fetches,
            self.run_config.receiver_adapter_instance,
            self.run_config.fetches_rating_n
            )

        run_start = time.time()
        measurement_tick = 0
        fetch_tick = 0
        for tick in range(self.run_config.max_ticks):
            printProgressBar(
                tick, self.run_config.max_ticks, prefix='Progress:', suffix='Complete', length=50)
            start = time.time()
            provider.tick(tick)
            if tick == fetch_tick:
                receiver.fetch(tick)
                # next tick with final measurement:
                fetch_tick = min(
                    tick + self.run_config.fetch_recommendations_skip_steps, self.run_config.max_ticks - 1)
            if tick == measurement_tick:
                # take measurement
                monitor.measure()
                # next tick with final measurement:
                measurement_tick = min(
                    tick + self.run_config.take_measurement_skip_steps, self.run_config.max_ticks - 1)
            end = time.time()
            run_time = end - start
            # put to sleep if faster then tick delay
            if run_time < self.run_config.tick_delay:
                sleep_time = max(self.run_config.tick_delay - run_time, 0.001)
                time.sleep(sleep_time)
        printProgressBar(self.run_config.max_ticks, self.run_config.max_ticks,
                         prefix='Progress:', suffix='Complete', length=50)

        run_end = time.time()
        print("Run finished in {0} ".format(run_end - run_start))
        # aggregating necessary data
        receiver.join_fetches()
        # shutdown all stuff
        provider.tear_down()
        receiver.tear_down()
        monitor.tear_down()
        return (run_start, run_end)

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
