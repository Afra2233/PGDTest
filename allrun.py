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
fig, ax = plt.subplots(figsize=(14, 14))

# 生成每个柱子的颜色
# colors = plt.cm.viridis(np.linspace(0, 1, len(avg_accuracies)))
# colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(avg_accuracies)))
colors = plt.get_cmap('tab20b')(np.linspace(0, 1, len(avg_accuracies)))
np.random.shuffle(colors) 

x_pos = np.arange(1, len(run_ids) + 1)
# 绘制柱状图
bars = ax.bar(x_pos, avg_accuracies, color=colors,tick_label=x_pos)
# line_offset = 0.5  # Adjust this value to control the distance above the bars

ax.plot(x_pos, avg_accuracies, color='green', linestyle='-', linewidth=1, label='Trend Line')



# 添加标题和轴标签
# ax.set_title('The Average Accuracy of General Classifier on All Features Per Run')
ax.set_xlabel('Run ID',fontsize=16)
ax.set_ylabel('Average General Classifier Accuracies',fontsize=16)

# 旋转x轴标签以便于阅读
plt.yticks(fontsize=16)
plt.xticks(rotation=45,fontsize=16)

# 显示图表
plt.show()
plt.savefig('General_Classifier.png')




dirty_clean_accuracy = {}

for stepsize, eps_clusters in selected_runs.items():
    for eps, runs_in_cluster in eps_clusters.items():
        for run in runs_in_cluster:
            dirty_on_clean_values = [value for key, value in run.summary.items() 
                                     if key.startswith("Test Dirty Classifier on Clean Features")]

            # 收集所有符合条件的 value - clean on dirty
            clean_on_dirty_values = [value for key, value in run.summary.items() 
                                     if key.startswith("Test Clean Classifier on Dirty Features")]

            # 计算 dirty on clean 的平均准确率
            if dirty_on_clean_values:
                average_dirty_on_clean = sum(dirty_on_clean_values) / len(dirty_on_clean_values)
                # dirty_on_clean_accuracy[run.id] = average_dirty_on_clean
            if clean_on_dirty_values:
                average_clean_on_dirty = sum(clean_on_dirty_values) / len(clean_on_dirty_values)
                # clean_on_dirty_accuracy[run.id] = average_clean_on_dirty
            total = average_clean_on_dirty+average_dirty_on_clean
            dirty_clean_accuracy[run.id] = total/2
            
dc_run_ids = []  # 存储所有运行的ID
dc_avg_accuracies = []

# 打印计算出的每个 run 的 dirty_clean_average_accuracy
for run_id, average_accuracy in dirty_clean_accuracy.items():
    print(f"Run ID: {run_id}, Dirty-Clean Average Accuracy: {average_accuracy:.3f}")
    dc_run_ids.append(run_id)
    dc_avg_accuracies.append(average_accuracy)

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(14, 14))

# 生成每个柱子的颜色
# colors = plt.cm.viridis(np.linspace(0, 1, len(dc_avg_accuracies)))
# colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(dc_avg_accuracies)))
# colors = plt.get_cmap('tab20b')(np.linspace(0, 1, len(dc_avg_accuracies)))
# np.random.shuffle(colors) 

x_pos = np.arange(1, len(dc_run_ids) + 1)
# 绘制柱状图
bars = ax.bar(x_pos, dc_avg_accuracies, color=colors,tick_label=x_pos)
# line_offset = 0.5  # Adjust this value to control the distance above the bars

ax.plot(x_pos, dc_avg_accuracies, color='green', linestyle='-', linewidth=1, label='Trend Line')



# 添加标题和轴标签
# ax.set_title('The Average Accuracy of Dirty Classifier on Clean Features and Clean Classifier on Drity Features Per Run')
ax.set_xlabel('Run ID',fontsize=16)
ax.set_ylabel('Average Dirty-Clean Classifier Accuracies',fontsize=16)

# 旋转x轴标签以便于阅读
plt.yticks(fontsize=16)
plt.xticks(rotation=45,fontsize=16)

# 显示图表
plt.show()
plt.savefig('DC_Classifier.png')


x_values = np.array(avg_accuracies)
y_values = np.array(dc_avg_accuracies)

markers = ['o', 's', 'D', '^', 'p']  # 圆形、方形、菱形、三角形、五边形
# colors = ['blue', 'green', 'purple', 'orange', 'red']

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制散点图
for i in range(len(x_values)):
    ax.scatter(x_values[i], y_values[i], color=colors[i], marker=markers[i % len(markers)], s=100)

# scatter = ax.scatter(x_values, y_values, color='blue', marker='o')

# 使用 np.polyfit 计算最佳拟合线的斜率和截距
slope, intercept = np.polyfit(x_values, y_values, 1)

# 生成拟合线的y值
fit_line = intercept + slope * x_values

# 绘制拟合线
# ax.plot(x_values, fit_line, label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}', color='red')
ax.plot(x_values, fit_line, label=f'Best Fit Line', color='red')
ax.legend()

# 添加标题和轴标签
ax.set_title('Comparison of Average Accuracies')
ax.set_xlabel('Average General Classifier Accuracies')
ax.set_ylabel('Average Dirty-Clean Classifier Accuracies')

# 添加每个点的标签
# for i, txt in enumerate(run_ids):
#     ax.annotate(txt, (avg_accuracies[i], dc_avg_accuracies[i]))

# 显示图表
plt.show()
plt.savefig('compare.png')

x_values = []
y_values = []
colors = []

# 遍历每个 run 并获取 train_eps 值
for run in runs:
    run_id = run.id
    if run_id in average_accuracies and run_id in dirty_clean_accuracy:
        avg_accuracy = average_accuracies[run_id]
        dirty_clean_avg_accuracy = dirty_clean_accuracy[run_id]

        # 提取 train_eps 参数并获取对应颜色
        train_eps = run.config.get("optimizer")
        color = optimizer_colors.get(train_eps, 'gray')  # 如果 train_eps 不在 eps_colors 中则用灰色

        # 存储数据点
        x_values.append(avg_accuracy)
        y_values.append(dirty_clean_avg_accuracy)
        colors.append(color)

# 转换为 numpy 数组
x_values = np.array(x_values).flatten()
y_values = np.array(y_values).flatten()

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制散点图，使用指定颜色
# ax.scatter(x_values, y_values, color=colors, marker='o', s=100,)
for optimizer, color in optimizer_colors.items():
    indices = [i for i in range(len(colors)) if colors[i] == color]
    ax.scatter(x_values[indices], y_values[indices], color=color, marker='o', s=100, label=f'optimizer={optimizer}')


# 使用 np.polyfit 计算最佳拟合线的斜率和截距
slope, intercept = np.polyfit(x_values, y_values, 1)
fit_line = intercept + slope * x_values

# 绘制拟合线
ax.plot(x_values, fit_line, label='Best Fit Line', color='red')
ax.legend()

# 添加标题和轴标签
ax.set_title('Comparison of Average Accuracies')
ax.set_xlabel('Average General Classifier Accuracies')
ax.set_ylabel('Average Dirty-Clean Classifier Accuracies')

# 显示图表并保存
plt.show()
plt.savefig('compare2.png')




