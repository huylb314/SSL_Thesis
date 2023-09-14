import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

# tags = ['critic/eval_1_critic_loss_MSE', 
#         'critic/eval_1_critic_loss', 
#         'critic/eval_10_critic_loss_MSE', 
#         'critic/eval_10_critic_loss',
#         'critic/eval_100_critic_loss_MSE',
#         'critic/eval_100_critic_loss',
#         'critic/eval_full_critic_loss_MSE', 
#         'critic/eval_full_critic_loss']

tags = [
        # 'critic/eval_1_critic_loss_MSE', 
        # 'critic/eval_1_critic_loss', 
        'critic/eval_10_critic_loss_MSE', 
        # 'critic/eval_10_critic_loss',
        # 'critic/eval_100_critic_loss_MSE',
        # 'critic/eval_100_critic_loss',
        'critic/eval_full_critic_loss_MSE', 
        # 'critic/eval_full_critic_loss'
        ]

# Step 1: Load the TensorBoard logs and extract the data
# logdir = '/home/leh2rng/log/'
# logdir = f"/home/leh2rng/log_epochs/pes_plot"
# logdir = f"/home/leh2rng/log_epochs/pes_500"
logdir = f"/home/leh2rng/log_epochs/log_plot"
save_dir = f"/home/leh2rng/log_epochs/"
weight = 0.8

def smooth(current_value, last_value, weight):  # Weight between 0 and 1
    smoothed_value = last_value * weight + (1 - weight) * current_value  # Calculate smoothed value
    return smoothed_value

def process_log(current_log_path):
    log_data = []
    last_value_tags = {}
    for tag in tags:
        last_value_tags[tag] = None
    if os.path.isdir(current_log_path):
        # Get the list of events files in the log directory3
        args = None
        method = None
        contexture = False
        events_files = [os.path.join(current_log_path, file) for file in os.listdir(current_log_path) if file.startswith('events.out')]
        args_file = os.path.join(current_log_path, "args.json")
        with open(args_file, 'r') as f:
            args = json.load(f)

        # parse name
        if "contexture" in args['ssl_method']:
            contexture = True
        if args['ssl_method'] == "online":
            # method = f"ONLINE_L{args['limit']}"
            # method = f"online_L{args['limit']}"
            method = f"online"
        elif args['w_online'] == 1 and args['w_pseudo'] == 1 and args['w_offline'] == 0:
            if args['ssl_method'] == "fixmatch":
                method = f"{args['ssl_method']}_L{args['limit']}_C{args['ssl_confidence']}"
            else:
                method = f"{args['ssl_method']}_L{args['limit']}_D{args['pseudo_samples']}_L{args['ssl_lower_confidence']}_W{args['weighted_method']}"
        modified_args = {k:int(v) if isinstance(v, bool) is True else v for k,v in args.items()}
        for file in events_files:
            for event in tf.compat.v1.train.summary_iterator(file):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        if value.tag in tags:
                            if last_value_tags[value.tag] is None:
                                last_value_tags[value.tag] = value.simple_value
                            smoothed_value = smooth(value.simple_value, last_value_tags[value.tag], weight) # Calculate smoothed value
                            log_data.append({**modified_args, 'epoch': event.step, 'tag': value.tag, 'value': value.simple_value, 'smoothed_value': smoothed_value, 'method': method, 'contexture': contexture})
                            last_value_tags[value.tag] = smoothed_value
    return log_data


def process_event(events, modified_args, method, contexture):
    # Process the event.summary.value here
    for event in events:
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag in tags:
                    data.append({**modified_args, 'epoch': event.step, 'tag': value.tag, 'value': value.simple_value, 'method': method, 'contexture': contexture})

def read_events(file):
    event_batches = []
    batch_size = 100  # Adjust this based on your use case

    for event in tf.compat.v1.train.summary_iterator(file):
        event_batches.append(event)
        if len(event_batches) >= batch_size:
            yield event_batches
            event_batches = []

    if event_batches:
        yield event_batches

# Read the TensorBoard logs and extract scalar data
data = []
num_processes = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=num_processes)
logdirs = [os.path.join(logdir, log) for log in os.listdir(logdir)]
for id, log_path in enumerate(tqdm(logdirs)):
    return_logs = pool.map(process_log, [log_path])
    for return_log in return_logs:
        data.extend(return_log)
pool.close()
pool.join()


# # Compute means and confidence intervals
# means = data.groupby('x')['y'].mean()
# ci_low = data.groupby('x')['y'].quantile(0.025)
# ci_high = data.groupby('x')['y'].quantile(0.975)

# # Find the x value with the minimum mean
# min_x = means.idxmin()
# min_mean = means[min_x]
# min_ci_low = ci_low[min_x]
# min_ci_high = ci_high[min_x]


df = pd.DataFrame(data)

means = df.groupby(['method', 'epoch', 'tag'])['smoothed_value'].mean()
# ci_low = df.groupby(['method', 'epoch', 'tag', 'method'])['value'].quantile(0.025)
# ci_high = df.groupby(['method', 'epoch', 'tag', 'method'])['value'].quantile(0.975)
stds = df.groupby(['method', 'epoch', 'tag'])['smoothed_value'].std()

ri_means = means.reset_index()
ri_stds = stds.reset_index()

ri_combine = ri_means.copy()
ri_combine['std'] = ri_stds['smoothed_value']

df_combine = ri_combine[ri_combine['epoch'] == 500]

ROUND_UP = 5

for tag in tags:
    print (tag)
    df_combine_selected = df_combine[(df_combine['tag'] == tag)]
    for i, row in df_combine_selected.iterrows():
        round_value = round(row["smoothed_value"], ROUND_UP)
        round_std = round(row["std"], ROUND_UP)
        print (f'{row["method"]} & {round_value} \pm {round_std}')

# filtered_ri_means = ri_means[(ri_means['epoch'] == 500) and (ri_means['tag'] == "critic/eval_10_critic_loss_MSE")]
# filtered_ri_stds = ri_stds[(ri_stds['epoch'] == 500) and (ri_stds['tag'] == "critic/eval_10_critic_loss_MSE")]

df_combine.to_excel(f"./df_combine_{time.time()}.xlsx")

# idx_min_means_x = means.idxmin()
# min_mean = means[idx_min_means_x]
# min_ci_low = ci_low[idx_min_means_x]
# min_ci_high = ci_high[idx_min_means_x]

# # means.groupby(['method']).min()

# dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss'])['value'].min()
# dfmin.to_excel(f"./dfmin_{time.time()}.xlsx")


# # Step 2: Convert the extracted data to a Pandas DataFrame
# # df.to_excel('./df.xlsx')
# dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss'])['value'].min()
# dfmin.to_excel(f"./dfmin_{time.time()}.xlsx")
