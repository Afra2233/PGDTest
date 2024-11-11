import wandb
api = wandb.Api()

ENTITY = "st7ma784"
PROJECT = "AllDataPGN"

# Initialize clusters with nested dictionaries for train_eps values
clusters = {
    1/255: {1/255: [], 2/255: [], 4/255: []},
    2/255: {1/255: [], 2/255: [], 4/255: []},
    4/255: {1/255: [], 2/255: [], 4/255: []}
}

runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    train_stepsize = run.config.get("train_stepsize")
    train_eps = run.config.get("train_eps")
    
    # if train_stepsize in clusters:
    #     if train_eps in clusters[train_stepsize]:
    #         clusters[train_stepsize][train_eps].append(run)
    if train_stepsize in clusters and train_eps in clusters[train_stepsize]:
        log_keys = run.history(keys=["_step", "Test General Classifier on Dirty Features on dataset 5 alpha * epsilon * step *"])
        for log in log_keys:
            key = "Test General Classifier on Dirty Features on dataset 5 alpha {} epsilon {} step {}".format(run.config.get("dataloader_idx"), train_stepsize, train_eps)
            if key in log:
                accuracy = log[key]
                clusters[train_stepsize][train_eps].append(accuracy)

# 计算每个分类的平均准确率
for stepsize, eps_dict in clusters.items():
    for eps, accuracies in eps_dict.items():
        if accuracies:  # 确保列表不为空
            avg_accuracy = sum(accuracies) / len(accuracies)
            average_accuracies[(stepsize, eps)] = avg_accuracy

# 数据准备
steps_eps = list(average_accuracies.keys())
accuracies = [average_accuracies[key] for key in steps_eps]

# 绘制曲线图
plt.figure(figsize=(10, 6))
for (stepsize, eps), accuracy in zip(steps_eps, accuracies):
    plt.plot(stepsize, accuracy, 'o-', label=f'stepsize {stepsize}, eps {eps}')

plt.xlabel('Train Stepsize and Epsilon')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy for Different Train Stepsize and Epsilon')
plt.legend()
plt.show()

# Print the runs organized by train_stepsize and then by train_eps
for stepsize, eps_clusters in clusters.items():
    print(f"Runs with train_stepsize {stepsize:.6f}:")
    for eps, runs_in_cluster in eps_clusters.items():
        print(f"  Eps {eps:.6f}: {len(runs_in_cluster)} runs")
        for run in runs_in_cluster:
            print(f"    Run ID: {run.id}, Name: {run.name}")

