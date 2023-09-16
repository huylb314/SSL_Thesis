# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# import os
# import tensorflow as tf
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import json
# from tqdm import tqdm
# import multiprocessing
# from functools import partial
# import time

# logdir = f"/home/leh2rng/amira_experiments/consac/log"
# realrobot_logs = [
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_online_12obj_1', 
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_online_12obj_2',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_flexmatch_softmax_12obj_1',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_flexmatch_softmax_12obj_2',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_fixmatch_12obj_1',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_fixmatch_12obj_2',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_flexmatch_12obj_1',
#     'MP_depth_smoothing_avg_norm_suctions6_bs4_flexmatch_12obj_2',
#     ]


# offline_picking_list = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # Step 1: Load the TensorBoard logs and extract the data
# save_dir = f"/home/leh2rng/log_epochs/"
# weight = 0.8
# SUCCESS_RATE_WINDOWS = 200
# BIN_CLEAR_RATE_WINDOWS = 5
# LIMIT = 1000

# # helpers
# def smooth(current_value, last_value, weight):  # Weight between 0 and 1
#     smoothed_value = last_value * weight + (1 - weight) * current_value  # Calculate smoothed value
#     return smoothed_value

# # Step 1: Load the TensorBoard logs and extract the data
# # logdir = '/home/leh2rng/log/'

# # Read the TensorBoard logs and extract scalar data
# data = []
# for id, realrobot_log_path in enumerate(tqdm(realrobot_logs)):
#     current_log_path = os.path.join(logdir, realrobot_log_path)
#     if os.path.isdir(current_log_path):
#         # Get the list of events files in the log directory3
#         parsed_name = realrobot_log_path.split("_")
#         method = parsed_name[7:-2]
#         method = " ".join(method)
#         seed = parsed_name[-1]
        
#         if method == "online":
#             method = method.capitalize()
#         elif method == "fixmatch":
#             method = "Fixmatch C0.95"
#         elif method == "flexmatch":
#             method = "Flexmatch C0.95 L0.9"
#         elif method == "contexture flexmatch softmax":
#             method = "Flexmatch Contexture Softmax C0.95 L0.9"    
        
#         args_file = os.path.join(current_log_path, "train_state.json")
#         with open(args_file, 'r') as f:
#             # print (f.readlines())
#             log_read = f.readlines()
#             latest_log = log_read[-1]
#             logs = json.loads(latest_log)

#         reward_list = logs['reward_list'][:LIMIT]
#         join_reward_list = offline_picking_list + reward_list
#         last_value = None
#         for idx_step in range(SUCCESS_RATE_WINDOWS, len(join_reward_list)):
#             success_rate_temp = join_reward_list[idx_step-SUCCESS_RATE_WINDOWS:idx_step]
#             success_rate_temp = sum(success_rate_temp) / SUCCESS_RATE_WINDOWS
#             if last_value is None:
#                 last_value = success_rate_temp
#             smoothed_value = smooth(success_rate_temp, last_value, weight) # Calculate smoothed value
#             # data.append({'Step': (idx_step-SUCCESS_RATE_WINDOWS) + 1, 'Success Rate': success_rate_temp, 'seed': seed, 'method': method})
#             data.append({'Step': (idx_step-SUCCESS_RATE_WINDOWS) + 1, 'Trailing Grasp Success Rate': smoothed_value, 'seed': seed, 'method': method})
#             last_value = smoothed_value

# # Step 2: Convert the extracted data to a Pandas DataFrame
# df = pd.DataFrame(data)

# # df.to_excel('./df.xlsx')
# # dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss', 'ssl_confidence'])['value'].min()
# # dfmin.to_excel(f"./dfmin_{time.time()}_FIXMATCH.xlsx")

# # Step 4: Use Seaborn to plot the data
# # sns.set_theme(style="darkgrid")
# sns.set_theme(style="whitegrid", font_scale=1.5)

# hue_order = ['Online', 'Fixmatch C0.95', 'Flexmatch C0.95 L0.9', 'Flexmatch Contexture Softmax C0.95 L0.9']
# colors = ['r', 'g', 'orange', 'b']

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_xlim(0, 1000)
# # ax.set_ylim(0.5, 1.0)
# # sns.lineplot(data=df, x='step', y='smoothed_value', hue='tag')
# # g_results = sns.lineplot(data=df, x="Step", y="Success Rate", hue="method", errorbar="sd")
# # g_results = sns.lineplot(data=df, x="Step", y="Trailing Grasp Success Rate", hue="method", errorbar="se")
# g_results = sns.lineplot(data=df, x="Step", y="Trailing Grasp Success Rate", hue_order=hue_order, hue="method", palette=colors, errorbar="se")
# # g_results.set(yscale='log')
# plt.legend(loc='lower right', fontsize="14").set_draggable(True)
# fig.savefig(f"{save_dir}/realrobot_w{weight}.pdf")
# plt.close()

# print ("#### DONE ####")





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

logdir = f"/home/leh2rng/amira_experiments/consac/log"
realrobot_logs = [
    'MP_depth_smoothing_avg_norm_suctions6_bs4_online_12obj_1', 
    'MP_depth_smoothing_avg_norm_suctions6_bs4_online_12obj_2',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_flexmatch_softmax_12obj_1',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_contexture_flexmatch_softmax_12obj_2',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_fixmatch_12obj_1',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_fixmatch_12obj_2',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_flexmatch_12obj_1',
    'MP_depth_smoothing_avg_norm_suctions6_bs4_flexmatch_12obj_2',
    ]


offline_picking_list = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Step 1: Load the TensorBoard logs and extract the data
save_dir = f"/home/leh2rng/log_epochs/"
weight = 0.8
SUCCESS_RATE_WINDOWS = 200
BIN_CLEAR_RATE_WINDOWS = 5
LIMIT = 1000

# helpers
def smooth(current_value, last_value, weight):  # Weight between 0 and 1
    smoothed_value = last_value * weight + (1 - weight) * current_value  # Calculate smoothed value
    return smoothed_value

# Step 1: Load the TensorBoard logs and extract the data
# logdir = '/home/leh2rng/log/'

# Read the TensorBoard logs and extract scalar data
data = []
for id, realrobot_log_path in enumerate(tqdm(realrobot_logs)):
    current_log_path = os.path.join(logdir, realrobot_log_path)
    if os.path.isdir(current_log_path):
        # Get the list of events files in the log directory3
        parsed_name = realrobot_log_path.split("_")
        method = parsed_name[7:-2]
        method = " ".join(method)
        seed = parsed_name[-1]
        
        if method == "online":
            method = method.capitalize()
        elif method == "fixmatch":
            method = "Fixmatch C0.95"
        elif method == "flexmatch":
            method = "Flexmatch C0.95 L0.9 FULL"
        elif method == "contexture flexmatch softmax":
            method = "Flexmatch Contexture Softmax C0.95 L0.9 FULL"    
        
        args_file = os.path.join(current_log_path, "train_state.json")
        with open(args_file, 'r') as f:
            # print (f.readlines())
            log_read = f.readlines()
            latest_log = log_read[-1]
            logs = json.loads(latest_log)
        
        print(method, seed, sum(logs['episode_bin_clear_list'][-15::])/15)
        
        bin_com = round(sum(logs['episode_bin_clear_list'][-15::])/15, 3)
        # reward_list = logs['reward_list'][:LIMIT]
        
        # print(method, seed, sum(reward_list[-(15*15)::])/(15*15))
        # bin_com = round(sum(reward_list[-(15*15)::])/(15*15), 2)
        
        data.append({'Bin Completion': bin_com, 'seed': seed, 'method': method})
        

df = pd.DataFrame(data)

print (df)
print (df.groupby(['method'])['Bin Completion'].mean())

# # df.to_excel('./df.xlsx')
# # dfmin = df.groupby(['tag', 'limit', 'method', 'weighted_method', 'use_pessimistic_loss', 'ssl_confidence'])['value'].min()
# # dfmin.to_excel(f"./dfmin_{time.time()}_FIXMATCH.xlsx")