import wandb
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import os
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

api = wandb.Api()

ENTITY = "st7ma784"
PROJECT = "AllDataPGN"

# Initialize clusters with nested dictionaries for train_eps values
clusters = {
    1/255: {1/255: [], 2/255: [], 4/255: []},
    2/255: {1/255: [], 2/255: [], 4/255: []},
    4/255: {1/255: [], 2/255: [], 4/255: []}
}
accuracies = {key: {subkey: [] for subkey in clusters[key]} for key in clusters}
# average_accuracies = {}
runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    train_stepsize = run.config.get("train_stepsize")
    train_eps = run.config.get("train_eps")
    
    if train_stepsize in clusters and train_eps in clusters[train_stepsize]:
        # Fetch logs and filter those relevant to the specified dataset
        for log in run.scan_history(keys=["_runtime", "_timestamp"]):
            if "Test General Classifier on Dirty Features on dataset 5" in log:
                acc_key = "Test General Classifier on Dirty Features on dataset {} alpha {} epsilon {} step {}".format(run.config['DataLoader_idx'], train_eps, run.config['train_stepsize'])
                if acc_key in log:
                    accuracies[train_stepsize][train_eps].append(log[acc_key])

# Calculate average accuracies and prepare data for plotting
plot_data = {key: np.mean(values) for key, values in accuracies.items() if values}

# Plotting
fig, ax = plt.subplots()
for stepsize, eps_dict in plot_data.items():
    ax.plot(list(eps_dict.keys()), list(eps_dict.values()), label=f'stepsize {stepsize}')
ax.set_xlabel('train_eps')
ax.set_ylabel('Average Accuracy')
ax.set_title('Average Accuracy vs. Train Step Size')
ax.legend()
plt.show()
# # Print the runs organized by train_stepsize and then by train_eps
# for stepsize, eps_clusters in clusters.items():
#     print(f"Runs with train_stepsize {stepsize:.6f}:")
#     for eps, runs_in_cluster in eps_clusters.items():
#         print(f"  Eps {eps:.6f}: {len(runs_in_cluster)} runs")
#         for run in runs_in_cluster:
#             print(f"    Run ID: {run.id}, Name: {run.name}")

