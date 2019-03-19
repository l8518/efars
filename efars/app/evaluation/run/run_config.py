import math
import os
import multiprocessing

class RunConfig():

    def __init__(self, plugins_module):
        # intial set up
        self.plugins_module = plugins_module
        self.run_prefix = "unknow"

        # random settings:
        self.seed = 123456789

        # Obtain monitor enviroment
        if os.getenv("DOCKER_CONTAINERS"):
            self.docker_containers = str(os.getenv("DOCKER_CONTAINERS")).split(",")
        else:
            self.docker_containers = ['basic-recommender', 'database']

        self.docker_sock = str(os.getenv("DOCKER_SOCK") or "unix://var/run/docker.sock" )

        # Hardcoded parameters
        self.data_root_folder = os.path.join("data")
        self.data_ratings_folder = os.path.join(self.data_root_folder, "ratings")
        self.runs_folder_path = os.path.join(self.data_root_folder, "runs")
        self.evaluation_plots_folder_path = os.path.join(self.data_root_folder , "evaluation_plots")
        self.evaluation_csv_folder_path = os.path.join(self.data_root_folder , "evaluation_csv")
        self.evaluation_test_dataset_filepath = os.path.join(self.data_ratings_folder, "test.csv")
        self.evaluation_source_file_delimiter = ","
        self.evaluation_source_file_newline = "\r\n"
        self.evaluation_pool_size = 5

        # Plan Parameters
        self.dataset_adapter = str(os.getenv("DATASET_ADAPTER") or 'MovieLensDatasetAdapter')

        # Run Parameters
        self.run_warm_up_seconds = int(os.getenv("RUN_WARM_UP_DELAY") or 30)
        self.training_source_file = str(os.getenv("RUN_TRAINING_FILE") or "./data/ratings/train.csv") 
        self.test_source_file = \
            str(os.getenv("RUN_TEST_FILE") or
                "./data/ratings/test.csv")
        self.run_n = int(os.getenv("RUN_N_REPETITION") or 1)
        self.provisions_per_tick = int(os.getenv("RUN_PROVISIONS_PER_TICK") or 2000)
        self.provider_adapter = str(os.getenv("RUN_PROVIDER_ADAPTER") or 'BasicRecommenderProviderAdapter')
        self.receiver_adapter = str(os.getenv("RUN_RECEIVER_ADAPTER") or 'BasicRecommenderReceiverAdapter')
        self.take_measurement_skip_steps = int(os.getenv("RUN_MEASURE_SKIP_STEPS") or 10)
        self.fetch_recommendations_skip_steps = int(os.getenv("RUN_FETCH_SKIP_STEPS") or 10)
        self.tick_delay = int(os.getenv("RUN_TICK_DELAY") or 2)
        self.concurrent_fetches = int(os.getenv("RUN_CONCURRENT_FETCHES") or multiprocessing.cpu_count())
        self.concurrent_provisions = int(os.getenv("RUN_CONCURRENT_PROVISIONS") or multiprocessing.cpu_count())
        self.fetches_rating_n = int(os.getenv("RUN_FETCHES_RATING_N") or 20)

        # Evaluation Parameters
        self.test_relevant_items_rating_threshold = 3.5
        self.test_split_ratio = 0.2
        
        self.streaming_count = None
        self.max_ticks = None

        # Construct plugins
        ProviderAdapter = getattr(self.plugins_module, self.provider_adapter )
        self.provider_adapter_instance = ProviderAdapter()
        ReceiverAdapter = getattr(self.plugins_module, self.receiver_adapter )
        self.receiver_adapter_instance = ReceiverAdapter()
        DatasetAdapter = getattr(self.plugins_module, self.dataset_adapter )
        self.dataset_adapter_instance = DatasetAdapter()
        

    def calc_run_metrics(self):
        # Run Parametes that are calculatred
        self.streaming_count = self.__get_file_rows_count(self.training_source_file)
        self.max_ticks = math.ceil(
            self.streaming_count / self.provisions_per_tick)

    def get_printable_config(self):
        result = ""
        result += "This file contains all relevant information, about the run configuration \n".format()
        result += "Prefix:\t {0}  \n".format(self.run_prefix)

        result += "#### Mointoring information #### \n".format()
        result += "Docker Socket:\t {0}  \n".format(self.docker_sock)
        result += "Monitored containers:\t {0}  \n".format(
            self.docker_containers)

        result += "#### Run Configuration #### \n".format()
        result += "Run Repetions (n):\t {0}  \n".format(self.run_n)
        result += "Total Emitted Ratings:\t {0}  \n".format(
            self.streaming_count)
        result += "Provisions per Tick:\t {0}  \n".format(
            self.provisions_per_tick)
        result += "Concurrent Provisions in Tick:\t {0}  \n".format(
            self.concurrent_provisions)
        result += "Maximum of Ticks:\t {0}  \n".format(
            self.max_ticks)
        result += "Ticks Delay (in s):\t {0}  \n".format(
            self.tick_delay)
        result += "Hardware Measurements Tick Rate:\t {0}  \n".format(
            self.take_measurement_skip_steps)
        result += "Fetch Recommendations Tick Rate:\t {0}  \n".format(
            self.fetch_recommendations_skip_steps)
        result += "Concurrent Fetches in Tick:\t {0}  \n".format(
            self.concurrent_fetches)
        result += "Recommendations per fetch:\t {0}  \n".format(
            self.fetches_rating_n)
        result += "Warm Up Duration (in s):\t {0}  \n".format(
            self.run_warm_up_seconds)
        return result

    def __get_file_rows_count(self, source_file_path):
        """ Counts the amount of lines in a file"""
        source_file = open(source_file_path)
        N = 0
        for line in source_file:
            N += 1
        source_file.close()
        return N
