import wandb
api = wandb.Api()

ENTITY = "st7ma784"  
PROJECT = "AllDataPGN"


clusters = {
    1/255: [],
    4/255: [],
    8/255: []
}
runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    alpha = run.config.get("train_stepsize")
    if alpha in clusters:
        clusters[alpha].append(run)

for alpha, runs_in_cluster in clusters.items():
    print(f"Runs with train_stepsize {alpha}:")
    for run in runs_in_cluster:
        print(f"  Run ID: {run.id}, Name: {run.name}")
