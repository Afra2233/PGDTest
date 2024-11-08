import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "as33xopq" 


api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

history = run.history()
# logs = history.filter(regex="test_dirty_batch_acc_.*")

# # 初始化一个空列表存储提取的数据
# data = []

# # 解析每行日志记录
# for index, row in logs.iterrows():
#     for col_name, value in row.items():
#         if pd.notna(value):
#             # 使用正则表达式提取 alpha, epsilon, numsteps 和 dataloader_idx
#             match = re.match(
#                 r"test_dirty_batch_acc_alpha_([\d.]+)_epsilon_([\d.]+)_numsteps_(\d+)/dataloader_idx_(\d+)",
#                 col_name
#             )
#             if match:
#                 alpha, epsilon, numsteps, dataloader_idx = match.groups()
#                 data.append({
#                     "alpha": float(alpha),
#                     "epsilon": float(epsilon),
#                     "numsteps": int(numsteps),
#                     "dataloader_idx": int(dataloader_idx),
#                     "test_accuracy": value
#                 })

# # 将数据转为 DataFrame
# df = pd.DataFrame(data)

# # 创建一个新的列，将 (alpha, epsilon) 组合作为字符串
# df['alpha_epsilon_pair'] = df.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)

# 映射 dataloader_idx 到数据集名称
dataset_mapping = {
    0: "cifar10",
    1: "cifar100",
    2: "STL10",
    3: "Food101",
    4: "dtd",
    5: "fgvc_aircraft",
    6: "tinyImageNet"
}

# # 替换 dataloader_idx 为数据集名称
# df['dataset'] = df['dataloader_idx'].map(dataset_mapping)

# # 检查数据格式是否正确
# print(df.head())

# # 设置绘图样式
# plt.figure(figsize=(14, 10))

# # 绘制测试准确率条形图，用颜色表示 dataloader_idx
# g = sns.barplot(
#     data=df,
#     x="alpha_epsilon_pair",
#     y="test_accuracy",
#     hue="dataset",
#     dodge=True,  # 将不同的 dataloader_idx 在每个 alpha_epsilon_pair 上分开显示
#     ci=None
#     # errorbar=None
# )

# # 添加图例和标签
# plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset")
# plt.xlabel("(Alpha, Epsilon) Pair")
# plt.ylabel("Test Accuracy")
# plt.xticks(rotation=90)
# plt.legend(title="Dataset")
# plt.tight_layout()

# # 显示图表
# plt.show()
# plt.savefig('test_acc.png')

logs_classfar = history.filter(regex="Test General Classifier on Dirty Features on dataset.*")

# 初始化一个空列表存储提取的数据
data_classifar = []

# 解析每行日志记录
for index, row in logs_classfar.iterrows():
    for col_name, value in row.items():
        if pd.notna(value):
            # 使用正则表达式提取 alpha, epsilon, numsteps 和 dataloader_idx
            # match = re.match(
            #     r"test_dirty_batch_acc_alpha_([\d.]+)_epsilon_([\d.]+)_numsteps_(\d+)/dataloader_idx_(\d+)",
            #     col_name
            # )
            match = re.match(
                r"Test General Classifier on Dirty Features on dataset (\d+) alpha ([\d.]+) epsilon ([\d.]+) step (\d+)", 
                col_name
            )
            if match:
                dataloader_idx, alpha, epsilon, numsteps = match.groups()
                data_classifar.append({
                    "alpha": float(alpha),
                    "epsilon": float(epsilon),
                    "numsteps": int(numsteps),
                    "dataloader_idx": int(dataloader_idx),
                    "test_accuracy": value
                })

# 将数据转为 DataFrame
df_classifar = pd.DataFrame(data_classifar)

# 创建一个新的列，将 (alpha, epsilon) 组合作为字符串
df_classifar['alpha_epsilon_pair'] = df_classifar.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)

# # 映射 dataloader_idx 到数据集名称
# dataset_mapping = {
#     0: "cifar10",
#     1: "cifar100",
#     2: "STL10",
#     3: "Food101",
#     4: "dtd",
#     5: "fgvc_aircraft",
#     6: "tinyImageNet"
# }

# # 替换 dataloader_idx 为数据集名称
dfdf_classifar['dataset'] = dfdf_classifar['dataloader_idx'].map(dataset_mapping)

# 检查数据格式是否正确
print(df_classifar.head())

# 设置绘图样式
plt.figure(figsize=(14, 10))

# 绘制测试准确率条形图，用颜色表示 dataloader_idx
g = sns.barplot(
    data=df_classifar,
    x="alpha_epsilon_pair",
    y="test_accuracy",
    hue="dataset",
    dodge=True,  # 将不同的 dataloader_idx 在每个 alpha_epsilon_pair 上分开显示
    ci=None
    # errorbar=None
)

# 添加图例和标签
plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset")
plt.xlabel("(Alpha, Epsilon) Pair")
plt.ylabel("Test classifiers's Accuracy")
plt.xticks(rotation=90)
plt.legend(title="Dataset")
plt.tight_layout()

# 显示图表
plt.show()
plt.savefig('test_classifiers_acc.png')

# logs_Test_Classifier = history.filter(regex="^Test General Classifier.*")

# # 初始化一个空列表存储符合条件的数据
# filtered_data = []
# for index, row in history.iterrows():
#     for col_name, value in row.items():
#         if pd.notna(value):
#             # 使用正则表达式筛选以 "Test General Classifier" 开头的日志
#             match = re.match(
#                 r"Test General Classifier on Dirty Features on dataset .* alpha ([\d.]+) epsilon ([\d.]+) step (\d+)", 
#                 col_name
#             )
#             if match:
#                 alpha, epsilon, step = match.groups()
#                 # 仅提取符合条件的记录
#                 if float(alpha) == 0.003921568859368563 and float(epsilon) == 0.003921568859368563 and int(step) == 9:
#                     filtered_data.append({
#                         "log_name": col_name,
#                         "value": value
#                     })

# # 将筛选后的数据打印出来
# for data in filtered_data:
#     print(f"{data['log_name']}: {data['value']}")



# # 过滤日志
# for index, row in logs_Test_Classifier.iterrows():
#     for col_name, value in row.items():
#         if pd.notna(value):
#             # 使用正则表达式筛选以 "Test General Classifier" 开头的日志
#             match = re.match(r"Test General Classifier on Dirty Features on dataset (\d+) alpha ([\d.]+) epsilon ([\d.]+) step (\d+)", col_name)
#             if match:
#                 dataloader_idx, alpha, epsilon, step = match.groups()
#                 # 仅提取符合 epsilon 前几位为 0.00392 且 step 为 9 的记录
#                 if epsilon.startswith("0.003921") and int(step) == 9:
#                     filtered_data.append({
#                         "dataset": dataloader_idx,
#                         "alpha": alpha,
#                         "epsilon": epsilon,
#                         "step": step,
#                         "test_accuracy": value
#                     })

# # 将数据转换为 DataFrame
# df = pd.DataFrame(filtered_data)

# # 将数据重塑为每个数据集为一列
# df_pivot = df.pivot(index="alpha", columns="dataset", values="test_accuracy")

# # 检查结果
# print(df_pivot)
