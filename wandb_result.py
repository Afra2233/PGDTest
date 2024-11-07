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
logs = history.filter(regex="test_dirty_batch_acc_.*")

# 初始化一个空列表存储提取的数据
data = []

# 解析每行日志记录
for index, row in logs.iterrows():
    for col_name, value in row.items():
        if pd.notna(value):
            # 使用正则表达式提取 alpha, epsilon, numsteps 和 dataloader_idx
            match = re.match(
                r"test_dirty_batch_acc_alpha_([\d.]+)_epsilon_([\d.]+)_numsteps_(\d+)/dataloader_idx_(\d+)",
                col_name
            )
            if match:
                alpha, epsilon, numsteps, dataloader_idx = match.groups()
                data.append({
                    "alpha": float(alpha),
                    "epsilon": float(epsilon),
                    "numsteps": int(numsteps),
                    "dataloader_idx": int(dataloader_idx),
                    "test_accuracy": value
                })

# 将数据转为 DataFrame
df = pd.DataFrame(data)

# 创建一个新的列，将 (alpha, epsilon) 组合作为字符串
df['alpha_epsilon_pair'] = df.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)

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

# 替换 dataloader_idx 为数据集名称
df['dataset'] = df['dataloader_idx'].map(dataset_mapping)

# 检查数据格式是否正确
print(df.head())

# 设置绘图样式
plt.figure(figsize=(14, 10))

# 绘制测试准确率条形图，用颜色表示 dataloader_idx
g = sns.barplot(
    data=df,
    x="alpha_epsilon_pair",
    y="test_accuracy",
    hue="dataset",
    dodge=True,  # 将不同的 dataloader_idx 在每个 alpha_epsilon_pair 上分开显示
    ci=None
)

# 添加图例和标签
plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset")
plt.xlabel("(Alpha, Epsilon) Pair")
plt.ylabel("Test Accuracy")
plt.xticks(rotation=90)
plt.legend(title="Dataset")
plt.tight_layout()

# 显示图表
plt.show()
plt.savefig('test_acc.png')

logs_Test_Classifier = history.filter(regex="^Test General Classifier.*")

# 打印结果
for index, row in logs_Test_Classifier.iterrows():
    for col_name, value in row.items():
        if col_name.startswith("Test General Classifier") and pd.notna(value):
            print(f"{col_name}: {value}")
