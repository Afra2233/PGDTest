import matplotlib.pyplot as plt

# 数据表中提供的数据
datasets = ["CIFAR10", "CIFAR100", "Food101", "DTD", "Tiny-ImageNet", "Average"]
epsilon_4_255 = [55.48, 46.51, 35.21, 3.78, 29.14, 34.02]
epsilon_4_255_LP = [86.82, 91.92, 68.27, 96.58, 99.61, 88.64]
epsilon_8_255 = [32.30, 28.90, 10.90, 6.67, 13.78, 22.05]
epsilon_8_255_LP = [78.77, 87.67, 49.04, 93.50, 98.63, 81.52]

# 绘制折线图
plt.figure(figsize=(10, 6))

# 使用不同的颜色表示不同的 epsilon 值
plt.plot(datasets, epsilon_4_255, marker='o', label=r'$\epsilon = 4/255$ (Ours)')
plt.plot(datasets, epsilon_4_255_LP, marker='o', label=r'$\epsilon = 4/255$ (Ours LP)')
plt.plot(datasets, epsilon_8_255, marker='o', label=r'$\epsilon = 8/255$ (Ours)')
plt.plot(datasets, epsilon_8_255_LP, marker='o', label=r'$\epsilon = 8/255$ (Ours LP)')

# 添加标签和标题
plt.xlabel("Dataset")
plt.ylabel("Adversarial Robust Accuracy")
plt.title("Adversarial Robust Accuracies across Datasets for Different $\epsilon$ Values")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
