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
        'critic/eval_full_critic_loss',
        'critic/eval_1_online_critic_f1_score_0',
        'critic/eval_1_online_critic_f1_score_1',
        'critic/eval_10_online_critic_f1_score_0',
        'critic/eval_10_online_critic_f1_score_1',
        'critic/eval_100_online_critic_f1_score_0',
        'critic/eval_100_online_critic_f1_score_1',
        'critic/eval_full_online_critic_f1_score_0',
        'critic/eval_full_online_critic_f1_score_1',
        ]

# Step 1: Load the TensorBoard logs and extract the data
logdir = f"/home/leh2rng/log_epochs/pes_500"
save_dir = f"/home/leh2rng/log_epochs/"

LOG_DIR = "/home/leh2rng/amira_experiments/consac/log"
weight = 0.8


plot_files = ["MP_depth_smoothing_avg_norm_suctions6_bs4_online_12obj_1", 
              "MP_depth_smoothing_avg_norm_suctions6_bs4_offline_12obj_1", 
              "MP_depth_smoothing_avg_norm_suctions6_bs4_fixmatch_12obj_1",
              "MP_depth_smoothing_avg_norm_suctions6_bs4_flexmatch_12obj_1",
              "MP_depth_smoothing_avg_norm_suctions6_bs4_freematch_12obj_1",
              "MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_flexmatch_12obj_1",
              "MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_freematch_12obj_1",
              ]

def smooth(current_value, last_value, weight):  # Weight between 0 and 1
    smoothed_value = last_value * weight + (1 - weight) * current_value  # Calculate smoothed value
    return smoothed_value

def process_log(current_log_path):
    log_data = []
    work_log = os.path.join(LOG_DIR, current_log_path)
    if os.path.isdir(work_log):
        # Get the list of events files in the log directory3
        args = None
        method = None
        args_file = os.path.join(work_log, "train_state.json")
        with open(args_file, 'r') as f:
            args = json.load(f)

        
        for success_rate in args['success_rate_list']:
            log_data.append({'step': event.step, 'tag': value.tag, 'value': value.simple_value, 'smoothed_value': smoothed_value, 'method': method})
        
        # parse name
        if args['ssl_method'] == "online":
            # method = f"ONLINE_L{args['limit']}"
            method = f"online_L{args['limit']}"
        elif args['w_online'] == 1 and args['w_pseudo'] == 1 and args['w_offline'] == 0:
            if args['ssl_method'] == "fixmatch":
                method = f"{args['ssl_method']}_L{args['limit']}_C{args['ssl_confidence']}"
            else:
                method = f"{args['ssl_method']}_L{args['limit']}_D{args['pseudo_samples']}_L{args['ssl_lower_confidence']}_W{args['weighted_method']}"
        
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
for id, log_path in enumerate(tqdm(plot_files)):
    return_logs = pool.map(process_log, [log_path])
    for return_log in return_logs:
        data.extend(return_log)
pool.close()
pool.join()

# Step 2: Convert the extracted data to a Pandas DataFrame
df = pd.DataFrame(data)

# df.to_excel('./df.xlsx')
# dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss', 'ssl_confidence'])['value'].min()
# dfmin.to_excel(f"./dfmin_{time.time()}_FIXMATCH.xlsx")

# Step 4: Use Seaborn to plot the data
sns.set_theme(style="darkgrid")

for tag in tags:
    df_selected = df.loc[(df['tag'] == tag)]

    df_selected = df_selected.sort_values(by ='method')

    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=df, x='step', y='smoothed_value', hue='tag')
    g_results = sns.lineplot(data=df_selected, x="step", y="smoothed_value", hue="method", errorbar="se")
    g_results.set(yscale='log')
    plt.legend(loc='best').set_draggable(True)
    name_tag = tag.replace("/", "_")
    fig.savefig(f"{save_dir}/{name_tag}_w{weight}.pdf")
    plt.close()


# for tag in tags:
#     for l in [100, 500]:
#         df_selected = df.loc[(df['limit'] == l) & (df['tag'] == tag)]
        
#         df_selected = df_selected.sort_values(by ='method')
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         # sns.lineplot(data=df, x='step', y='smoothed_value', hue='tag')
#         g_results = sns.lineplot(data=df_selected, x="step", y="smoothed_value", hue="method", errorbar="se")
#         g_results.set(yscale='log')
#         plt.legend(loc='best').set_draggable(True)
#         name_tag = tag.replace("/", "_")
#         fig.savefig(f"{save_dir}/limit{l}_{name_tag}_w{weight}.pdf")
#         plt.close()



        
print ("#### DONE ####")