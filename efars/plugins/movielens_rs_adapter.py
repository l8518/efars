import pandas
from app.planning.planner import DatasetAdapter
import os
import pathlib


class MovieLensRSDatasetAdapter(DatasetAdapter):
    """
    The MovieLensDatasetAdapter splits a
    MovieLens Ratings File into test and train set
    based on random sampling.
    """

    def __init__(self):
        """ Initializes a new MovieLensDatasetAdapter that uses random sampling """
        super().__init__()

    def generate(self, source_file_path, target_folder, seed, test_split_ratio,
                 rating_threshold, max_user_n):
        """ Generates a test and train dataset based on the largest user
         profiles and and a 50% user-based split"""

        pathlib.Path(target_folder).mkdir(
            parents=True, exist_ok=True)

        test_filepath = os.path.join(target_folder, "test.csv")
        train_filepath = os.path.join(target_folder, "train.csv")
        testr_filepath = os.path.join(target_folder, "test_relevant_items.csv")
        testir_filepath = os.path.join(
            target_folder, "test_irrelevant_items.csv")

        # Load MovieLens Dataset
        df = pandas.read_csv(source_file_path)
        df_len = len(df)

        # Split every selected user into 50% test and train set
        test_df = df.sample(frac=0.2, random_state=seed)

        # Remove all ratings selected from the whole set
        # to obtain the training dataset
        trainset = pandas.read_csv(source_file_path).drop(test_df.index)
        len_tdf = len(test_df)
        len_trainset = len(trainset)
        print("Ratings in Train Dataset: {0} ".format(len_trainset))
        print("Ratings in Test Dataset: {0}".format(len_tdf))
        print("Ratio of Test Dataset: {0}".format(len_tdf / df_len))
        print(
            "Amount of Ratings in whole Train + Test Datasets: {0}"
            .format(len_tdf + len_trainset))
        print("Amount of Ratings in whole Dataset: {0}".format(df_len))

        # Additionally calculate the relevant and irrelevant sets
        # for the test set
        relevant_items = test_df.loc[test_df['rating'].astype(
            'float64') >= rating_threshold]
        irrelevant_items = test_df.loc[test_df['rating'].astype(
            'float64') < rating_threshold]

        # Shuffle the trainset by sampling the whole dataset
        # as MovieLens ratings file is sorted by user id
        trainset = trainset.sample(frac=1.0, random_state=seed)
        # write to csv
        trainset.to_csv(train_filepath, header=False, index=False)
        test_df.to_csv(test_filepath, header=False, index=False)
        relevant_items.to_csv(testr_filepath, header=False, index=False)
        irrelevant_items.to_csv(testir_filepath, header=False, index=False)


if __name__ == '__main__':
    mlds = MovieLensDatasetAdapter()
    mlds.generate("./data/ml-20m/ratings.csv",
                  "./data/ratings/", 290219, 0.2, 3.5, 1700)
