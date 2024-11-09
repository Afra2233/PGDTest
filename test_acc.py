import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from fractions import Fraction

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "1h2lllig" 
# RUN_ID = "olnxxo2t"
# f5jq1b65 1h2lllig
# RUN_ID = "as33xopq"


api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

dataset_mapping = {
    0: "cifar10",
    1: "cifar100",
    2: "STL10",
    3: "Food101",
    4: "dtd",
    5: "fgvc_aircraft",
    6: "tinyImageNet"
}


# history = run.history()
# logs = history.filter(regex="test_dirty_batch_acc_.*")

summary_data = run.summary
logs_yy = {key: value for key, value in summary_data.items() if key.startswith("test_dirty_batch_acc_")}

# 初始化一个空列表存储提取的数据
data = []
for col_name, value in logs_yy.items():
    if pd.notna(value):
        # 使用正则表达式提取 alpha, epsilon, numsteps 和 dataloader_idx
        match = re.match(
            r"test_dirty_batch_acc_alpha_([\d.]+)_epsilon_([\d.]+)_numsteps_(\d+)/dataloader_idx_(\d+)",
            col_name
        )
        if match:
            alpha, epsilon, numsteps,dataloader_idx = match.groups()
            data.append({
                "alpha": float(alpha),
                 "epsilon": float(epsilon),
                 "numsteps": int(numsteps),
                 "dataloader_idx": int(dataloader_idx),
                 "test_accuracy": value
                })

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
print(data)

# 将数据转为 DataFrame
df = pd.DataFrame(data)

# # 创建一个新的列，将 (alpha, epsilon) 组合作为字符串
df['alpha_epsilon_pair_long'] = df.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)
df['alpha_epsilon_pair'] = df.apply(lambda row: f"({Fraction(row['alpha']).limit_denominator()}, {Fraction(row['epsilon']).limit_denominator()})", axis=1)



# 替换 dataloader_idx 为数据集名称
df['dataset'] = df['dataloader_idx'].map(dataset_mapping)

# 检查数据格式是否正确
print(df.head())

# 设置绘图样式
plt.figure(figsize=(14, 14))

# 绘制测试准确率条形图，用颜色表示 dataloader_idx
g = sns.barplot(
    data=df,
    x="alpha_epsilon_pair",
    y="test_accuracy",
    hue="dataset",
    dodge=True,  # 将不同的 dataloader_idx 在每个 alpha_epsilon_pair 上分开显示
    # ci=None
    errorbar=None
)

# 添加图例和标签
plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset (Without Linear Prob)",fontsize=24)
plt.xlabel("(Alpha, Epsilon) Pair",fontsize=24)
plt.ylabel("Test Accuracy",fontsize=24)
plt.xticks(rotation=90,fontsize=24)
plt.yticks(fontsize=24)
plt.legend(title="Dataset",fontsize=24)
plt.tight_layout()

# # 显示图表
plt.show()
plt.savefig('test_acc2.png')
