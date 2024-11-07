import wandb

wandb.login()


ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"  
RUN_ID = "795d73up" 


api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")


test_results = run.summary.get("test_results")  # 替换为你实际的测试结果字段名称


print("Test Results:", test_results)
