import wandb
api = wandb.Api()

project = api.project("your-entity/your-project-name")

runs = project.runs()


clusters = {
    1/255: [],
    4/255: [],
    8/255: []
}

for run in runs:
    alpha = run.config.get("train_stepsize")
    if alpha in clusters:
        clusters[alpha].append(run)

for alpha, runs_in_cluster in clusters.items():
    print(f"Runs with train_stepsize {alpha}:")
    for run in runs_in_cluster:
        print(f"  Run ID: {run.id}, Name: {run.name}")
