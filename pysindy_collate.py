import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import sys

def sort_by_start_and_merge(df):
    df = df.sort_values(by=['start'])
    
    last_start = None
    last_stop = None
    last_index = None
    for row in df.itertuples():
        #check if this time interval overlaps with the last one
        if last_start is not None and row.start <= last_stop:
            #merge the intervals
            last_stop = max(last_stop, row.stop)
            df.at[last_index, 'start'] = last_start
            df.at[last_index, 'stop'] = last_stop
            #drop the current row
            df = df.drop(row.Index)
            last_start = df.at[last_index, 'start']
            last_stop = df.at[last_index, 'stop']
            last_index = last_index
        else:
            last_start = row.start
            last_stop = row.stop
            last_index = row.Index

    return df.reset_index(drop=True)



def check_for_time_reverse(df):
    a = df['start'].to_numpy()
    b = df['stop'].to_numpy()
    c = a[1:] - b[:-1]

    problems = np.where(c < 0)
    return len(problems[0]) > 0

def compute_preictal_windows(df, pre_ictal_duration=25):
    a = df['start'].to_numpy()
    b = df['stop'].to_numpy()
    c = a[1:] - b[:-1]

    sig_indices = np.where(c > pre_ictal_duration)[0]

    starts = a[sig_indices + 1]
    starts = np.insert(starts, 0, a[0])
    starts = starts[:-1]
    stops = b[sig_indices]
    return starts, stops

def create_ictal_entries(starts, stops):
    f = np.ones(starts.shape[0]) * 2
    ictal_times = pd.DataFrame({'ictal_start' : starts,'ictal_stop': stops, 'ictal_id': f} )
    ictal_times.reset_index(drop=True)
    return ictal_times

def create_preictal_entries(ictal_times, pre_ictal_duration=25):
    pre_ictal_start= ictal_times['ictal_start'].to_numpy() - pre_ictal_duration
    pre_ictal_end = ictal_times['ictal_start'].to_numpy()
    pre_ictal_id = np.ones(len(pre_ictal_start)) * 1
    pre_ictal_times = pd.DataFrame({'pre_ictal_start' : pre_ictal_start,'pre_ictal_stop': pre_ictal_end, 'pre_ictal_id': pre_ictal_id} )
    return pre_ictal_times


def create_interictal_entries(ictal_times, pre_ictal_times):

    interictal_start = ictal_times['ictal_stop'].to_numpy()[:-1] 

    interictal_stop = pre_ictal_times['pre_ictal_start'].to_numpy()[1:]
    interictal_id = np.ones(len(interictal_start)) * 0


    interictal_times = pd.DataFrame({'interictal_start' : interictal_start,'interictal_stop': interictal_stop, 'interictal_id': interictal_id})
    return interictal_times

def create_combined_df(ictal_times, pre_ictal_times, interictal_times):

    # Relabel the columns
    ictal_times.columns = ['epoch_start', 'epoch_stop', 'epoch_id']  # Replace with your desired column names
    pre_ictal_times.columns = ['epoch_start', 'epoch_stop', 'epoch_id']
    interictal_times.columns = ['epoch_start', 'epoch_stop', 'epoch_id']    

    # Assuming the three dataframes are named df1, df2, and df3
    # Combine all dataframes into one
    combined_df = pd.concat([ictal_times, pre_ictal_times, interictal_times], ignore_index=True)

    # Sort the rows by the first column in ascending order
    sorted_df = combined_df.sort_values(by='epoch_start').reset_index(drop=True)

    return sorted_df



def find_pattern_break_index(vector):
    pattern = np.array([1, 2, 0])
    pattern_length = len(pattern)
    
    for i in range(0, len(vector) // len(pattern) , pattern_length):
        if not np.array_equal(vector[i:i+pattern_length], pattern):
            return i  # Return the starting index of the mismatched pattern
    return -1  # Return -1 if the pattern is not broken


def check_for_pattern_break(sorted_df):

    break_index = find_pattern_break_index(sorted_df['epoch_id'].to_numpy())

    if break_index == -1:
        print("The vector maintains the pattern throughout.")
        return False
    else:
        print(f"The pattern breaks at index {break_index}.")
        return True



def create_mini_epochs(sorted_df,mini_epoch_length=25000):

    start_times = []
    stop_times = []
    epoch_ids = []
    for row in tqdm(sorted_df.itertuples()):
        start_time = row.epoch_start
        end_time = row.epoch_stop
        start_index = int(start_time * 5000)
        end_index = int(end_time * 5000)
        
        epoch_sample_duration = end_index - start_index + 1

        for i in range(0,mini_epoch_length*(epoch_sample_duration//mini_epoch_length),mini_epoch_length):
            start_times.append(i/5000 + start_time)
            stop_times.append((i+mini_epoch_length)/5000 + start_time)
            epoch_ids.append(row.epoch_id)


    exp_df = pd.DataFrame({'start_time':start_times,'stop_time':stop_times,'epoch_id':epoch_ids})

    return exp_df


def collate_data(annotation_file, output_file):

    df = pd.read_csv(annotation_file)

    sorted_df = sort_by_start_and_merge(df)

    if check_for_time_reverse(sorted_df):
        print("Warning: Time reversals detected in the data.")
        sys.exit()
         
    else:
        print("No time reversals detected.")
        

    ictal_starts, ictal_stops = compute_preictal_windows(sorted_df)
    ictal_times = create_ictal_entries(ictal_starts, ictal_stops)
    pre_ictal_times = create_preictal_entries(ictal_times)
    interictal_times = create_interictal_entries(ictal_times, pre_ictal_times)

    combined_df = create_combined_df(ictal_times, pre_ictal_times, interictal_times)

    if  check_for_pattern_break(combined_df):
        print("Warning: The expected pattern of epoch IDs is broken.")
        sys.exit()
    else:        print("The expected pattern of epoch IDs is maintained.")

    mini_epoch_df = create_mini_epochs(combined_df)

    with open(output_file,'wb') as file: 
        pickle.dump(mini_epoch_df,file)


if __name__ == "__main__":
    annotation_file = '/app/Data/Annotations/AD_annotation_log_20260217_202718_DRAFT.csv'
    output_file = '/app/Data/WR/Annotations/260223_PYSINDy_annotations.pkl'
    
    collate_data(annotation_file, output_file)