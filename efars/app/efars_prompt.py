from cmd import Cmd
import os
from dotenv import load_dotenv
from app.planning import Planner
from app.evaluation import Run, RunConfig
from app.comparison import Comparator, Plotter


class EfarsPrompt(Cmd):
    def __init__(self, plugins_module):
        # Read and Set Config:
        load_dotenv()
        self.plugins_module = plugins_module
        self.run_config = RunConfig(plugins_module)
        # Initialize other components:
        self.planner = Planner()
        # Customize shell
        self.prompt = 'efars> '
        self.intro = StaticContent.get_intro()
        super(EfarsPrompt, self).__init__()

    def do_plan(self, inp):
        'Planning of the evaluation, downloads required dataset and splits accordingly.'

        question = ""
        question += "Which dataset to you want to use? \n"
        question += "(1) MovieLens Latest Datasets (small) \n"
        question += "(2) MovieLens 20m Datasets (recommended for research) \n"
        question += "\n"
        question += "Type in number to download, or none to  cancel: "
        file_selection = input(question)

        download_file = ""
        if file_selection == "1":
            download_file = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            extracted_file = "./data/ml-latest-small/ratings.csv"
            user_n = 5  # TODO, this requires a setting
        elif file_selection == "2":
            download_file = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
            extracted_file = "./data/ml-20m/ratings.csv"
            user_n = 1700  # TODO, this requires a setting
        else:
            print("Unknow option")
            return
        print("downloading test data")
        self.planner.download(download_file)
        print("preparing test data")
        self.run_config.dataset_adapter_instance.generate(
            extracted_file, "./data/ratings/", self.run_config.seed, self.run_config.test_split_ratio, self.run_config.test_relevant_items_rating_threshold, user_n)

    def do_run(self, inp):
        'Runs the evaluation based on the current configuration specified by enviroment and default values.'
        # construct necessary plugins:
        self.run_config.calc_run_metrics()
        question = ""
        question += "Add a prefix to this evaluation, otherwise default will be used (and data overwritten!): \n"
        prefix = input(question)
        if prefix == "":
            prefix = "unnamed"
        else:
            prefix = "{0}".format(prefix)
        self.run_config.run_prefix = prefix

        self.run = Run(self.run_config)
        self.run.start()

    def do_print_config(self, inp):
        'Prints the current evaluation configuration.'
        print(self.run_config.get_printable_config())

    def do_assessment(self, inp):
        'Performs all necessary assements for the persisted calculations.'
        evl = Comparator(self.run_config)
        print("This might take a while")
        evl.precalc_fetch_data()
        evl.precalc_monitor_data()
        evl.precalc_provider_data()
        print("metrics calculated...")
        self.do_compare(inp)

    def do_compare(self, inp):
        'Plots the relevant diagrams for comparision.'
        pltr = Plotter(self.run_config)
        pltr.load_fetch_csvs()
        pltr.load_monitor_csvs()
        pltr.load_provider_csvs()
        pltr.calculate_quality_metrics_results()
        pltr.calculate_performance_results()
        pltr.calculate_performance_boxplots()
        pltr.calculate_quality_metrics_boxplots()

    def do_exit(self, inp):
        'Closes EFARS.'
        print("efars is closing, this might take a while...")
        return True

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)

        print("Unknown command: {}".format(inp))


class StaticContent:

    @staticmethod
    def get_intro():
        intro = ""
        intro += "####### #######    #    ######   #####  \n"
        intro += "#       #         # #   #     # #     # \n"
        intro += "#       #        #   #  #     # #       \n"
        intro += "#####   #####   #     # ######   #####  \n"
        intro += "#       #       ####### #   #         # \n"
        intro += "#       #       #     # #    #  #     # \n"
        intro += "####### #       #     # #     #  #####  \n"
        intro += "--------------------------------------------------------- \n"
        intro += "EFARS - Evaluation Framework for Architectures of Recommender Systems \n\n"
        intro += "Please type ? to list commands. \n "
        return intro
