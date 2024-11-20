import wandb
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import os
import pandas as pd
from utils import cosine_lr
from utils import one_hot_embedding
from utils import accuracy,clamp,normalize
import torch.nn.functional as F
from clip import clip
from models.prompters import TokenPrompter, NullPrompter
from torchattacks import AutoAttack
from utils import clip_img_preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
#get default dict for logging
from collections import defaultdict
import threading
import time
import matplotlib.pyplot as plt
import queue
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline
api = wandb.Api()

ENTITY = "st7ma784"
PROJECT = "AllDataPGN"

# Initialize clusters with nested dictionaries for train_eps values
clusters = {
    1/255: {1/255: [], 2/255: [], 4/255: []},
    2/255: {1/255: [], 2/255: [], 4/255: []},
    4/255: {1/255: [], 2/255: [], 4/255: []}
}
eps_colors = {1/255: 'blue', 2/255: 'green', 4/255: 'purple'}
# learning_rate", default=5e-4, options=[5e-5,5e-4,1e-5]
lr_colors ={5e-4:'blue',5e-4:'green',1e-5:'purple'}
# "sgd","adam","adamw"
optimizer_colors ={"sgd": 'blue', "adam": 'green', "adamw": 'purple'}
train_numsteps_colors = {5:'blue',10:'green'}

accuracies = {key: {subkey: [] for subkey in clusters[key]} for key in clusters}
# average_accuracies = {}
runs = api.runs(f"{ENTITY}/{PROJECT}")
import wandb

api = wandb.Api()



selected_runs = {}

runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    train_stepsize = run.config.get("train_stepsize")
    train_eps = run.config.get("train_eps")

    if train_stepsize in clusters and train_eps in clusters[train_stepsize]:
        clusters[train_stepsize][train_eps].append(run)
        

for stepsize, eps_clusters in clusters.items():
    print(f"Runs with train_stepsize {stepsize:.6f}:")
    for eps, runs_in_cluster in eps_clusters.items():
        print(f"  Eps {eps:.6f}: {len(runs_in_cluster)} runs")
        for run in runs_in_cluster:
            has_relevant_key = any(key.startswith("Test General Classifier on Dirty Features on dataset 1") for key in run.summary.keys())
            if has_relevant_key:          
                   
                if stepsize not in selected_runs:
                    selected_runs[stepsize] = {}
      
                if eps not in selected_runs[stepsize]:
                    selected_runs[stepsize][eps] = []
                # selected_runs[stepsize][eps].append(run)
                selected_runs[stepsize][eps].append(run)
                


for stepsize, eps_clusters in selected_runs.items():
    print(f"Selected Runs with train_stepsize {stepsize:.6f}:")
    for eps, selected_runs_in_cluster in eps_clusters.items():
        print(f"  Selected Eps {eps:.6f}: {len(selected_runs_in_cluster)} runs")
        for run in selected_runs_in_cluster:
            print(f"  Run ID: {run.id}")
            print(f"    Train Step Size: {stepsize:.6f}, Train EPS: {eps:.6f}")

average_accuracies = {}
max_accuracy = 0
best_run_info = None

for stepsize, eps_clusters in selected_runs.items():
    for eps, runs_in_cluster in eps_clusters.items():
        for run in runs_in_cluster:
            # accuracies = [value for key, value in run.summary.items() if key.startswith("Test General Classifier on All Features")]
            accuracies = [value for key, value in run.summary.items() if key.startswith("Test")]
           
            if accuracies:  
                average_accuracy = sum(accuracies) / len(accuracies)
                
                # if average_accuracy > max_accuracy:
                #     max_accuracy = average_accuracy
                #     best_run_info = (run.id, run.name, average_accuracy)
                # if run.id not in average_accuracies:
                #     average_accuracies[run] = []
                # average_accuracies[run].append(average_accuracy)
                average_accuracies[run] = average_accuracy



for run_id, accuracies in average_accuracies.items():
    print(f"Run ID: {run_id}, Average Accuracies: {accuracies}")
run_ids = []  
avg_accuracies = []  

data_ic = []
for run, avg_accuracies in average_accuracies.items():
    # avg_accuracy = np.mean(avg_accuracies)  # 平均accuracy
    config = run.config
    data_ic.append({
        "learning_rate": config.get("learning_rate", None),
        "train_eps": config.get("train_eps", None),
        "train_numsteps": config.get("train_numsteps", None),
        "train_stepsize": config.get("train_stepsize", None),
        "test_numsteps": config.get("test_numsteps", None),
        "test_stepsize": config.get("test_stepsize", None),
        "test_eps": config.get("test_eps", None),
        "optimizer": config.get("optimizer", None),
        # "train_numsteps": config.get("train_number", None),
        "precision": config.get("precision", None),
        "accuracy": avg_accuracies
    })


df_ic = pd.DataFrame(data_ic)

df_ic["optimizer"] = df_ic["optimizer"].astype("category").cat.codes


X = df_ic[["learning_rate", "train_eps", "train_numsteps", "train_stepsize", "test_numsteps", "test_stepsize", "test_eps", "optimizer","precision"]]
Y = df_ic["accuracy"]


rf = RandomForestRegressor(random_state=10)
rf.fit(X, Y)

feature_importances = rf.feature_importances_

correlations = []
for col in X.columns:
    rho, _ = spearmanr(df_ic[col], Y)
    correlations.append(rho)


results = pd.DataFrame({
    "Parameter": X.columns,
    "Importance": feature_importances,
    "Correlation": correlations
})
results = results.sort_values(by="Importance", ascending=False).reset_index(drop=True)

print(results)

# import ace_tools as tools
# tools.display_dataframe_to_user("Parameter Importance and Correlation Table", results)


# for run_id, accuracies in average_accuracies.items():
#     for acc in accuracies:
#         run_ids.append(run_id)
#         avg_accuracies.append(acc)



# fig, ax = plt.subplots(figsize=(14, 14))


# # colors = plt.cm.viridis(np.linspace(0, 1, len(avg_accuracies)))
# # colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(avg_accuracies)))
# colors = plt.get_cmap('tab20b')(np.linspace(0, 1, len(avg_accuracies)))
# np.random.shuffle(colors) 

# x_pos = np.arange(1, len(run_ids) + 1)

# bars = ax.bar(x_pos, avg_accuracies, color=colors,tick_label=x_pos)
# # line_offset = 0.5  # Adjust this value to control the distance above the bars

# # ax.plot(x_pos, avg_accuracies, color='green', linestyle='-', linewidth=2, label='Trend Line')

# # smooth_x = np.linspace(x_pos.min(), x_pos.max(), 300)  # More points for smoother line
# # smooth_y = make_interp_spline(x_pos, avg_accuracies)(smooth_x)  # Interpolated values


# # ax.plot(smooth_x, smooth_y, color='purple', linestyle='-', linewidth=3, alpha=0.7, label='Trend Line')



# # ax.set_title('The Average Accuracy of General Classifier on All Features Per Run')
# ax.set_xlabel('Run ID',fontsize=18)
# ax.set_ylabel('Average General Classifier Accuracies',fontsize=18)


# plt.yticks(fontsize=18)
# plt.xticks(rotation=45,fontsize=18)


# plt.show()
# plt.savefig('General_Classifier.png')




# dirty_clean_accuracy = {}

# for stepsize, eps_clusters in selected_runs.items():
#     for eps, runs_in_cluster in eps_clusters.items():
#         for run in runs_in_cluster:
#             dirty_on_clean_values = [value for key, value in run.summary.items() 
#                                      if key.startswith("Test Dirty Classifier on Clean Features")]

           
#             clean_on_dirty_values = [value for key, value in run.summary.items() 
#                                      if key.startswith("Test Clean Classifier on Dirty Features")]

       
#             if dirty_on_clean_values:
#                 average_dirty_on_clean = sum(dirty_on_clean_values) / len(dirty_on_clean_values)
#                 # dirty_on_clean_accuracy[run.id] = average_dirty_on_clean
#             if clean_on_dirty_values:
#                 average_clean_on_dirty = sum(clean_on_dirty_values) / len(clean_on_dirty_values)
#                 # clean_on_dirty_accuracy[run.id] = average_clean_on_dirty
#             total = average_clean_on_dirty+average_dirty_on_clean
#             dirty_clean_accuracy[run.id] = total/2
            
# dc_run_ids = []  
# dc_avg_accuracies = []


# for run_id, average_accuracy in dirty_clean_accuracy.items():
#     print(f"Run ID: {run_id}, Dirty-Clean Average Accuracy: {average_accuracy:.3f}")
#     dc_run_ids.append(run_id)
#     dc_avg_accuracies.append(average_accuracy)


# fig, ax = plt.subplots(figsize=(14, 14))


# # colors = plt.cm.viridis(np.linspace(0, 1, len(dc_avg_accuracies)))
# # colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(dc_avg_accuracies)))
# # colors = plt.get_cmap('tab20b')(np.linspace(0, 1, len(dc_avg_accuracies)))
# # np.random.shuffle(colors) 

# x_pos = np.arange(1, len(dc_run_ids) + 1)

# bars = ax.bar(x_pos, dc_avg_accuracies, color=colors,tick_label=x_pos)
# # line_offset = 0.5  # Adjust this value to control the distance above the bars

# # ax.plot(x_pos, dc_avg_accuracies, color='green', linestyle='-', linewidth=2, label='Trend Line')
# # smooth_x = np.linspace(x_pos.min(), x_pos.max(), 300)  # More points for smoother line
# # smooth_y = make_interp_spline(x_pos, dc_avg_accuracies)(smooth_x)  # Interpolated values


# # ax.plot(smooth_x, smooth_y, color='purple', linestyle='-', linewidth=3, alpha=0.7, label='Trend Line')





# # ax.set_title('The Average Accuracy of Dirty Classifier on Clean Features and Clean Classifier on Drity Features Per Run')
# ax.set_xlabel('Run ID',fontsize=16)
# ax.set_ylabel('Average Dirty-Clean Classifier Accuracies',fontsize=16)


# plt.yticks(fontsize=16)
# plt.xticks(rotation=45,fontsize=16)


# plt.show()
# plt.savefig('DC_Classifier.png')


# x_values = np.array(avg_accuracies)
# y_values = np.array(dc_avg_accuracies)

# markers = ['o', 's', 'D', '^', 'p']  
# # colors = ['blue', 'green', 'purple', 'orange', 'red']


# fig, ax = plt.subplots(figsize=(10, 10))


# for i in range(len(x_values)):
#     ax.scatter(x_values[i], y_values[i], color=colors[i], marker=markers[i % len(markers)], s=100)

# # scatter = ax.scatter(x_values, y_values, color='blue', marker='o')

# slope, intercept = np.polyfit(x_values, y_values, 1)


# fit_line = intercept + slope * x_values

# # ax.plot(x_values, fit_line, label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}', color='red')
# ax.plot(x_values, fit_line, label=f'Best Fit Line', color='red')
# ax.legend()


# ax.set_title('Comparison of Average Accuracies')
# ax.set_xlabel('Average General Classifier Accuracies')
# ax.set_ylabel('Average Dirty-Clean Classifier Accuracies')


# # for i, txt in enumerate(run_ids):
# #     ax.annotate(txt, (avg_accuracies[i], dc_avg_accuracies[i]))


# plt.show()
# plt.savefig('compare.png')

# x_values = []
# y_values = []
# colors = []


# for run in runs:
#     run_id = run.id
#     if run_id in average_accuracies and run_id in dirty_clean_accuracy:
#         avg_accuracy = average_accuracies[run_id]
#         dirty_clean_avg_accuracy = dirty_clean_accuracy[run_id]

       
#         train_eps = run.config.get("optimizer")
#         color = optimizer_colors.get(train_eps, 'gray')  

 
#         x_values.append(avg_accuracy)
#         y_values.append(dirty_clean_avg_accuracy)
#         colors.append(color)


# x_values = np.array(x_values).flatten()
# y_values = np.array(y_values).flatten()


# fig, ax = plt.subplots(figsize=(10, 10))


# # ax.scatter(x_values, y_values, color=colors, marker='o', s=100,)
# for optimizer, color in optimizer_colors.items():
#     indices = [i for i in range(len(colors)) if colors[i] == color]
#     ax.scatter(x_values[indices], y_values[indices], color=color, marker='o', s=100, label=f'optimizer={optimizer}')



# slope, intercept = np.polyfit(x_values, y_values, 1)
# fit_line = intercept + slope * x_values


# ax.plot(x_values, fit_line, label='Best Fit Line', color='red')
# ax.legend()


# ax.set_title('Comparison of Average Accuracies')
# ax.set_xlabel('Average General Classifier Accuracies')
# ax.set_ylabel('Average Dirty-Clean Classifier Accuracies')


# plt.show()
# plt.savefig('compare2.png')




