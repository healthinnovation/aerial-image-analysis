import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

if __name__ == "__main__":
    '''
    [StratifIAD] Script that divides data into N groups. Each group will form a fold for the 
    cross-validation (leave-one-fold-out) and cross-testing. To divide the data into N groups, 
    we first shuffle the data. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_csv", type=str, help="This is the csv of the folders where the class is present", required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_exp", type=int, help="This is the number of the experiment. The division of data is different from each experiment", required=True)

    args = parser.parse_args()
    #df = pd.DataFrame({'class': os.listdir(args.data_csv)})
    #print(type(df))
    #print(df)
    #df2 = pd.read_csv ('/home/ubuntu/cgiar-earthobservation/data/buildings_folders.csv')
    #df2 = df2.drop('number',1)
    #print(type(df2))
    #print(df2)

    df = pd.read_csv (args.data_csv)
    df = df.drop('number',1)
    #df = pd.DataFrame({'crop': os.listdir(args.data_dir)})

    output_dir = '../data/experiment_'+f'{args.num_exp:03}/'
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    trainfile = f'{output_dir}train_00.csv'
        
    print(f'Saving train/dev dataset #00')
    # df.to_csv(trainfile, index=False)
    kf = KFold(n_splits = args.num_folds, shuffle = False, random_state = None)
    cv = 0
    for train_index, test_index in kf.split(df):
        print(f'Saving train/dev dataset #00 --> CV {cv}')

        trainfile = f'{output_dir}train_00_cv_{cv:02}.csv'
        devfile = f'{output_dir}dev_00_cv_{cv:02}.csv'

        # print("TRAIN:", train_index, "TEST:", test_index)
        train, dev = df.iloc[train_index], df.iloc[test_index]
            
        train.to_csv(trainfile, index=False)
        dev.to_csv(devfile, index=False)
        cv += 1
