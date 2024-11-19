import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from fractions import Fraction

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "f5jq1b65" 


api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
print(run.config)

dataset_mapping = {
    0: "cifar10",
    1: "cifar100",
    2: "STL10",
    3: "Food101",
    4: "dtd",
    5: "fgvc_aircraft",
    6: "tinyImageNet"
}


history = run.history()
# logs = history.filter(regex="test_dirty_batch_acc_.*")


# data = []


# for index, row in logs.iterrows():
#     for col_name, value in row.items():
#         if pd.notna(value):
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


# df = pd.DataFrame(data)


# df['alpha_epsilon_pair_long'] = df.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)
# df['alpha_epsilon_pair'] = df.apply(lambda row: f"({Fraction(row['alpha']).limit_denominator()}, {Fraction(row['epsilon']).limit_denominator()})", axis=1)



#
# df['dataset'] = df['dataloader_idx'].map(dataset_mapping)


# print(df.head())


# plt.figure(figsize=(14, 14))


# g = sns.barplot(
#     data=df,
#     x="alpha_epsilon_pair",
#     y="test_accuracy",
#     hue="dataset",
#     dodge=True,  
#     # ci=None
#     errorbar=None
# )


# plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset (Without Linear Prob)",fontsize=24)
# plt.xlabel("(Alpha, Epsilon) Pair",fontsize=24)
# plt.ylabel("Test Accuracy",fontsize=24)
# plt.xticks(rotation=90,fontsize=24)
# plt.yticks(fontsize=24)
# plt.legend(title="Dataset",fontsize=24)
# plt.tight_layout()


# plt.show()
# plt.savefig('test_acc.png')



summary_data = run.summary

logs_yy = {key: value for key, value in summary_data.items() if key.startswith("Test General Classifier on")}


# for col_name, value in logs_yy.items():
#     print(f"{col_name}: {value}")




# linear_filtered_data =[]
# for col_name, value in summary_data.items():
#     if pd.notna(value):
#        
#         match = re.match(
#             r"Test General Classifier on Dirty Features on dataset (\d+) alpha (0\.00392[\d]*) epsilon (0\.0392[\d]*) step (9)", 
#             col_name
#         )
#         if match:
#             dataloader_idx, alpha, epsilon, numsteps = match.groups()
#             linear_filtered_data.append({
#                 "alpha": float(alpha),
#                 "epsilon": float(epsilon),
#                 "numsteps": int(numsteps),
#                 "dataloader_idx": int(dataloader_idx),
#                 "test_accuracy": value
#             })


# for item in linear_filtered_data:
#     print(item)



data_classifar = []

for col_name, value in logs_yy.items():
    if pd.notna(value):
        
        # match = re.match(
        #     r"test_dirty_batch_acc_alpha_([\d.]+)_epsilon_([\d.]+)_numsteps_(\d+)/dataloader_idx_(\d+)",
        #     col_name
        # )
        match = re.match(
            r"Test General Classifier on Clean Features on dataset (\d+) alpha ([\d.]+) epsilon ([\d.]+) step (\d+)", 
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


df_classifar = pd.DataFrame(data_classifar)



df_classifar['alpha_epsilon_pair_long'] = df_classifar.apply(lambda row: f"({round(row['alpha'], 6)}, {round(row['epsilon'], 6)})", axis=1)
df_classifar['alpha_epsilon_pair'] = df_classifar.apply(lambda row: f"({Fraction(row['alpha']).limit_denominator()}, {Fraction(row['epsilon']).limit_denominator()})", axis=1)



df_classifar['dataset'] = df_classifar['dataloader_idx'].map(dataset_mapping)


# if 'alpha_epsilon_pair' in df_classifar.columns:
#    
#     filtered_rows = df_classifar[df_classifar['alpha_epsilon_pair'].str.contains(r'\(1/255')]
#     print(filtered_rows)
# else:
#     print("DataFrame does not contain the required 'alpha_epsilon_pair' column.")

pd.set_option('display.max_rows', None)    # Show all rows
# pd.set_option('display.max_columns', None) # Show all columns

print(df_classifar)


plt.figure(figsize=(14, 14))


g = sns.barplot(
    data=df_classifar,
    x="alpha_epsilon_pair",
    y="test_accuracy",
    hue="dataset",
    dodge=True,  
    # ci=None
    errorbar=None
)


# plt.title("Test Accuracy by (Alpha, Epsilon) Pair and Dataset (With Linear prob)",fontsize=24)
plt.xlabel("(Alpha, Epsilon) Pair",fontsize=24)
plt.ylabel("Test Accuracy",fontsize=24)
plt.xticks(rotation=90,fontsize=24)
plt.yticks(fontsize=24)
plt.legend(title="Dataset",fontsize=24)
plt.tight_layout()
# plt.ylim(50, 100)
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y * 100)))

plt.show()
plt.savefig('test_classifiers_acc.png')
