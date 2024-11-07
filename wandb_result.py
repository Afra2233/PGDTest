import wandb

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "795d73up" 


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

# 检查数据格式是否正确
print(df.head())

# 设置绘图样式
sns.set(style="whitegrid")

# 绘制测试准确率散点图，用颜色表示 alpha，用形状表示 epsilon
plt.figure(figsize=(12, 8))
g = sns.scatterplot(
    data=df,
    x="dataloader_idx",
    y="test_accuracy",
    hue="alpha",
    style="epsilon",
    palette="viridis",
    s=100
)

# 添加图例和标签
plt.title("Test Accuracy by Dataset Index, Alpha, and Epsilon")
plt.xlabel("Dataset (dataloader_idx)")
plt.ylabel("Test Accuracy")
plt.legend(title="Alpha and Epsilon")
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.show()
