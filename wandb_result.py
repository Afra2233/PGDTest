import wandb

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "795d73up" 


api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

history = run.history()
logs = history.filter(regex="test_dirty_batch_acc_alpha_.*")
for index, row in logs.iterrows():
    print(row.to_dict())
# test_results = run.summary.get("test_results")  # 替换为你实际的测试结果字段名称


# print("Test Results:", test_results)
