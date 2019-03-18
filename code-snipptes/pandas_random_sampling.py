import pandas
import os
import pathlib

if __name__ == '__main__':
    # load dataset
    file_path = "../efars/data/ml-20m/ratings.csv"
    seed = 1
    df = pandas.read_csv(file_path)
    df_grouped = df[['userId', 'movieId']] \
        .groupby(['userId']).count()

    # get 20% fraction of dataset
    sampled_df = df.sample(frac=0.2, random_state=seed)
    # group count userIds
    sampled_df_grouped = sampled_df[['userId', 'movieId']] \
        .groupby(['userId']).count()

    print("Length of Dataset : {0}".format(len(df)))
    print("Amount of Users in Dataset : {0}"
          .format(len(df_grouped)))
    print("Amount of Users in 20% Random Sample: {0}"
          .format(len(sampled_df_grouped)))

# Outputs:
# Length of Dataset : 20000263
# Amount of Users in Dataset : 138493
# Amount of Users in 20% Random Sample: 138351
