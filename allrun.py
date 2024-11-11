import wandb
api = wandb.Api()

ENTITY = "st7ma784"
PROJECT = "AllDataPGN"

# Initialize clusters with nested dictionaries for train_eps values
clusters = {
    1/255: {1/255: [], 4/255: [], 8/255: []},
    2/255: {1/255: [], 4/255: [], 8/255: []},
    4/255: {1/255: [], 4/255: [], 8/255: []}
}

runs = api.runs(f"{ENTITY}/{PROJECT}")
for run in runs:
    train_stepsize = run.config.get("train_stepsize")
    train_eps = run.config.get("train_eps")
    
    if train_stepsize in clusters:
        if train_eps in clusters[train_stepsize]:
            clusters[train_stepsize][train_eps].append(run)

# Print the runs organized by train_stepsize and then by train_eps
for stepsize, eps_clusters in clusters.items():
    print(f"Runs with train_stepsize {stepsize:.6f}:")
    for eps, runs_in_cluster in eps_clusters.items():
        print(f"  Eps {eps:.6f}: {len(runs_in_cluster)} runs")
        for run in runs_in_cluster:
            print(f"    Run ID: {run.id}, Name: {run.name}")

