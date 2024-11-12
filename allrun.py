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
import wandb

api = wandb.Api()

ENTITY = "st7ma784"
PROJECT = "AllDataPGN"


selected_runs = {}

runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    train_stepsize = run.config.get("train_stepsize")
    train_eps = run.config.get("train_eps")

    if train_stepsize in clusters and train_eps in clusters[train_stepsize]:
        clusters[train_stepsize][train_eps].append(run)
        # 检查 summary 中的特定记录
        # for key in run.summary:
        #     if key.startswith("Test General Classifier on Dirty Features on dataset 1"):
        #         # 存储符合条件的 run 及其配置信息
        #         if train_stepsize not in selected_runs:
        #             selected_runs[train_stepsize] = {}
        #         if train_eps not in selected_runs[train_stepsize]:
        #             selected_runs[train_stepsize][train_eps] = []
        #         selected_runs[train_stepsize][train_eps].append(run)

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
                

# 打印选定的 runs 信息
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
            accuracies = [value for key, value in run.summary.items() if key.startswith("Test General Classifier on All Features")]
           
            if accuracies:  
                average_accuracy = sum(accuracies) / len(accuracies)
                
                if average_accuracy > max_accuracy:
                    max_accuracy = average_accuracy
                    best_run_info = (run.id, run.name, average_accuracy)
                if run.id not in average_accuracies:
                    average_accuracies[run.id] = []
                average_accuracies[run.id].append(average_accuracy)


# 现已更新和计算的 average_accuracies 字典
for run_id, accuracies in average_accuracies.items():
    print(f"Run ID: {run_id}, Average Accuracies: {accuracies}")
run_ids = []  # 存储所有运行的ID
avg_accuracies = []  # 存储对应的平均准确率

# 遍历 average_accuracies 字典收集数据
for run_id, accuracies in average_accuracies.items():
    for acc in accuracies:
        run_ids.append(run_id)
        avg_accuracies.append(acc)


# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 10))

# 生成每个柱子的颜色
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_accuracies)))

# 绘制柱状图
bars = ax.bar(run_ids, avg_accuracies, color=colors)


# 添加标题和轴标签
ax.set_title('The Average Accuracy of General Classifier on All Features Per Run')
ax.set_xlabel('Run ID')
ax.set_ylabel('Average Accuracy')

# 旋转x轴标签以便于阅读
plt.xticks(rotation=45)

# 显示图表
plt.show()
plt.savefig('General_Classifier.png')

# fig, ax = plt.subplots()


# for stepsize, eps_accuracies in average_accuracies.items():
#     for eps, accuracies in eps_accuracies.items():

#         ax.plot([stepsize]*len(accuracies), accuracies, label=f'Eps {eps:.6f}', marker='o')


# ax.set_xlabel('Train StepSize(alpha)')
# ax.set_ylabel('Average Accuracy')
# ax.set_title('Average Accuracy vs. alpha')
# ax.legend()

# plt.show()
# plt.savefig('average_accuracy.png')
# if best_run_info:
#     print(f"Best Run ID: {best_run_info[0]}, Name: {best_run_info[1]}, Average Accuracy: {best_run_info[2]:.6f}")

# # ax.set_xlabel('Average Accuracy')
# # ax.set_ylabel('Train Step Size')
# ax.set_title('Average Accuracy vs. Train Step Size')
# ax.legend()
# plt.show()
# plt.savefig('average_accuracy.png')
# # Print the runs organized by train_stepsize and then by train_eps
# for stepsize, eps_clusters in clusters.items():
#     print(f"Runs with train_stepsize {stepsize:.6f}:")
#     for eps, runs_in_cluster in eps_clusters.items():
#         print(f"  Eps {eps:.6f}: {len(runs_in_cluster)} runs")
#         for run in runs_in_cluster:
#             print(f"    Run ID: {run.id}, Name: {run.name}")

