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
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

sns.set(font_scale=2)

tags = ['critic/eval_1_critic_loss_MSE', 
        # 'critic/eval_1_critic_loss', 
        'critic/eval_10_critic_loss_MSE', 
        # 'critic/eval_10_critic_loss',
        'critic/eval_100_critic_loss_MSE',
        # 'critic/eval_100_critic_loss',
        'critic/eval_full_critic_loss_MSE', 
        # 'critic/eval_full_critic_loss',
        # 'critic/eval_1_online_critic_f1_score_0',
        # 'critic/eval_1_online_critic_f1_score_1',
        # 'critic/eval_10_online_critic_f1_score_0',
        # 'critic/eval_10_online_critic_f1_score_1',
        # 'critic/eval_100_online_critic_f1_score_0',
        # 'critic/eval_100_online_critic_f1_score_1',
        # 'critic/eval_full_online_critic_f1_score_0',
        # 'critic/eval_full_online_critic_f1_score_1',
        ]

# Step 1: Load the TensorBoard logs and extract the data
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
        contexture = ""
        events_files = [os.path.join(current_log_path, file) for file in os.listdir(current_log_path) if file.startswith('events.out')]
        args_file = os.path.join(current_log_path, "args.json")
        with open(args_file, 'r') as f:
            args = json.load(f)

        # parse name
        if args['pseudo_samples'] == -1:
            args['pseudo_samples'] = 'Full'
        if "contexture" in args['ssl_method']:
            contexture = " Contexture"
        new_name = args['ssl_method'].replace("contexture_", "")
        if args['ssl_method'] == "online":
            # method = f"ONLINE_L{args['limit']}"
            method = f"Online {args['limit']}"
            # method = f"online"
        elif args['w_online'] == 1 and args['w_pseudo'] == 1 and args['w_offline'] == 0:
            if args['ssl_method'] == "fixmatch":
                # method = f"{args['ssl_method']}_L{args['limit']}_C{args['ssl_confidence']}"
                # method = f"{args['ssl_method'].capitalize()} C{args['ssl_confidence']}"
                method = f"{args['ssl_method'].capitalize()}"
            else:
                if args['weighted_method'] == "no_weighted":
                    # method = f"{args['ssl_method']}_L{args['limit']}_D{args['pseudo_samples']}_L{args['ssl_lower_confidence']}_W{args['weighted_method']}"
                    # method = f"{new_name.capitalize()}{contexture} C{args['ssl_confidence']} L{args['ssl_lower_confidence']} {args['pseudo_samples']}"
                    method = f"{new_name.capitalize()}{contexture} L{args['ssl_lower_confidence']}"
                else:
                    method = f"{new_name.capitalize()}{contexture} Softmax L{args['ssl_lower_confidence']}"
        modified_args = {k:int(v) if isinstance(v, bool) is True else v for k,v in args.items()}
        for file in events_files:
            for event in tf.compat.v1.train.summary_iterator(file):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        if value.tag in tags:
                            if last_value_tags[value.tag] is None:
                                last_value_tags[value.tag] = value.simple_value
                            smoothed_value = smooth(value.simple_value, last_value_tags[value.tag], weight) # Calculate smoothed value
                            log_data.append({**modified_args, 'Epoch': event.step, 'tag': value.tag, 'value': value.simple_value, 'MSE': smoothed_value, 'method': method, 'contexture': contexture})
                            last_value_tags[value.tag] = smoothed_value
    return log_data


def process_event(events, modified_args, method, contexture):
    # Process the event.summary.value here
    for event in events:
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag in tags:
                    data.append({**modified_args, 'Epoch': event.step, 'tag': value.tag, 'value': value.simple_value, 'method': method, 'contexture': contexture})

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
# dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss', 'ssl_confidence'])['value'].min()
# dfmin.to_excel(f"./dfmin_{time.time()}_FIXMATCH.xlsx")

# Step 4: Use Seaborn to plot the data
# sns.set_theme(style="darkgrid")
sns.set_theme(style="whitegrid", font_scale=1.5)
# sns.set(font_scale = 1.5)

for tag in tags:
    df_selected = df.loc[(df['tag'] == tag)]

    df_selected = df_selected.sort_values(by ='method', ascending=False)
    hue_order = ['Online 500', 'Online 3000', 'Fixmatch']
    
    unique_name = df_selected['method'].unique()
    # unique_name = np.delete(unique_name, -1)
    # unique_name = np.delete(unique_name, -1)
    # unique_name = np.delete(unique_name, -1)
    unique_name = list(unique_name)
    
    hue_order = hue_order + unique_name
    # hue_order = set(hue_order)
    # hue_order = list(hue_order)
    
    colors = sns.color_palette(n_colors=len(unique_name))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 500)
    # sns.lineplot(data=df, x='step', y='smoothed_value', hue='tag')
    # g_results = sns.lineplot(data=df_selected, x="Epoch", y="MSE", hue_order=hue_order, hue="method", palette=['r', 'g'], errorbar="se")
    # g_results = sns.lineplot(data=df_selected, x="Epoch", y="MSE", hue_order=hue_order, hue="method", palette=colors, errorbar="se")
    
    g_results = sns.lineplot(data=df_selected, x="Epoch", y="MSE", hue_order=hue_order, hue="method", palette=colors, errorbar="se")
    # g_results = sns.lineplot(data=df_selected, x="Epoch", y="MSE", hue_order=hue_order, hue="method", errorbar="se")
    # g_results = sns.lineplot(data=df_selected, x="Epoch", y="MSE", hue="method", errorbar="se")
    # g_results = sns.lineplot(data=df_selected, x="epoch", y="smoothed_value", hue="method", errorbar="ci")
    g_results.set(yscale='log')
    plt.legend(loc='upper right', fontsize="12").set_draggable(True)
    # g_results.tick_params(labelsize=20)
    name_tag = tag.replace("/", "_")
    # g_results.set_xlabel(fontsize=30)
    # g_results.set_ylabel(fontsize=20)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    fig.savefig(f"{save_dir}/{name_tag}_w{weight}_contexture.pdf")
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