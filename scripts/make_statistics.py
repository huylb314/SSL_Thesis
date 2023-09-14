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

tags = ['critic/eval_1_critic_loss_MSE', 
        'critic/eval_1_critic_loss', 
        'critic/eval_10_critic_loss_MSE', 
        'critic/eval_10_critic_loss',
        'critic/eval_100_critic_loss_MSE',
        'critic/eval_100_critic_loss',
        'critic/eval_full_critic_loss_MSE', 
        'critic/eval_full_critic_loss']

# Step 1: Load the TensorBoard logs and extract the data
# logdir = '/home/leh2rng/log/'
LOWER_BOUND = 0.9
logdir = f"/home/leh2rng/log_save/{LOWER_BOUND}"
logdir = f"/home/leh2rng/log_save/{LOWER_BOUND}"

def process_log(current_log_path):
    log_data = []
    if os.path.isdir(current_log_path):
        # Get the list of events files in the log directory3
        args = None
        method = None
        events_files = [os.path.join(current_log_path, file) for file in os.listdir(current_log_path) if file.startswith('events.out')]
        args_file = os.path.join(current_log_path, "args.json")
        with open(args_file, 'r') as f:
            args = json.load(f)

        # parse name
        if args['w_online'] == 1 and args['w_pseudo'] == 0 and args['w_offline'] == 0:
            method = "ONLINE"
        elif args['w_online'] == 1 and args['w_pseudo'] == 1 and args['w_offline'] == 0:
            method = f"{args['ssl_method']}_D{args['pseudo_samples']}"
        modified_args = {k:int(v) if isinstance(v, bool) is True else v for k,v in args.items()}
        for file in events_files:
            for event in tf.compat.v1.train.summary_iterator(file):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        if value.tag in tags:
                            log_data.append({**modified_args, 'step': event.step, 'tag': value.tag, 'value': value.simple_value, 'method': method})
    return log_data


def process_event(events, modified_args, method, contexture):
    # Process the event.summary.value here
    for event in events:
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag in tags:
                    data.append({**modified_args, 'step': event.step, 'tag': value.tag, 'value': value.simple_value, 'method': method, 'contexture': contexture})

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

# Step 2: Convert the extracted data to a Pandas DataFrame
df = pd.DataFrame(data)
# df.to_excel('./df.xlsx')
dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss'])['value'].min()
dfmin.to_excel(f"./dfmin_{time.time()}_{LOWER_BOUND}.xlsx")
