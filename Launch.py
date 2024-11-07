
import os,sys
import pytorch_lightning
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping, ModelCheckpoint
import datetime
from pytorch_lightning.plugins.environments import SLURMEnvironment
from models.trainPGD import myLightningModule

#### This is our launch function, which builds the dataset, and then runs the model on it.


def train(config={
        "batch_size":64, # ADD MODEL ARGS HERE
         "codeversion":"-1",
    },dire=None,devices=None,accelerator=None,Dataset=None,logtool=None):


    model=myLightningModule(**config)
    if dire is None:
        dire=config.get("root",".")
    if Dataset is None:
        from DataModule import MyDataModule
        Dataset=MyDataModule(Cache_dir=dire,**config)
    elif config.get("dataset",None)== 'coco':
        from COCODataModule import MyDataModule
        Dataset=MyDataModule(Cache_dir=dire,**config)
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    filename="model-{}".format(model.version)
    modelSavedir=config.get("model_dir",dire)
   
    callbacks=[
        TQDMProgressBar(),
        # EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
        ModelCheckpoint(monitor="train_loss", mode="min",dirpath=modelSavedir,filename=filename,save_top_k=1,save_last=True,save_weights_only=True),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    print("Launching with precision",p)
    #hi
    #workaround for NCCL issues on windows
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    trainer=pytorch_lightning.Trainer(
            devices="auto" if devices is None else devices,
            accelerator="auto",
            max_epochs=config.get("epochs",10),
            inference_mode=False,
            #profiler="advanced",
            #plugins=[SLURMEnvironment()],
            #https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
            logger=logtool,
            # strategy=FSDPStrategy(accelerator="gpu",
            #                        parallel_devices=6,
            #                        cluster_environment=SLURMEnvironment(),
            #                        timeout=datetime.timedelta(seconds=1800),
            #                        #cpu_offload=True,
            #                        #mixed_precision=None,
            #                        #auto_wrap_policy=True,
            #                        #activation_checkpointing=True,
            #                        #sharding_strategy='FULL_SHARD',
            #                        #state_dict_type='full'
            # ),
           

            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            precision=p,
            fast_dev_run=config.get("debug",False),
    )
    if not os.path.exists(os.path.join(modelSavedir,filename+".ckpt")):
    # if not os.path.exists(os.path.join(modelSavedir,filename)):

            trainer.fit(model,Dataset)
    else:
            # model = model.load_from_checkpoint(os.path.join(modelSavedir,filename))
            model=myLightningModule.load_from_checkpoint(os.path.join(modelSavedir,filename+".ckpt"))
            trainer.test(model,Dataset)

    # trainer.test(model,Dataset)

#### This is a wrapper to make sure we log with Weights and Biases, You'll need your own user for this.
def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):

    USER="st7ma784"  
    PROJECT="AllDataPGN"
    NAME="TestDeploy"
    import pytorch_lightning
    import wandb
    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        wandb.login(key=os.getenv("WANDB_API_KEY","9cf7e97e2460c18a89429deed624ec1cbfb537bc")) 
        logtool= pytorch_lightning.loggers.WandbLogger(project=PROJECT,entity=USER, save_dir=os.getenv("WANDB_CACHE_DIR","."))                               #<-----CHANGE ME
        print(config)

    else:
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        print("Would recommend changing projectname according to config flags if major version swithching happens")
        try:
            run=wandb.init(project=PROJECT,entity=USER,name=NAME,config=config)                                          
            #check if api key exists in os.environ
        except:
            if "WANDB_API_KEY" not in os.environ:
                if "wandb" in os.environ:
                    os.environ["WANDB_API_KEY"]=os.environ["wandb"]
                    wandb.login(key=os.getenv("WANDB_API_KEY","9cf7e97e2460c18a89429deed624ec1cbfb537bc")) 
                else:
                    print("No API key found, please set WANDB_API_KEY in environment variables")
            wandb.login(key=os.getenv("WANDB_API_KEY","9cf7e97e2460c18a89429deed624ec1cbfb537bc"))

            run=wandb.init(project=PROJECT,entity=USER,name=NAME,config=config)                                           #<-----CHANGE ME      

        #os.environ["WANDB_API_KEY"]="9cf7e97e2460c18a89429deed624ec1cbfb537bc"  
        logtool= pytorch_lightning.loggers.WandbLogger(project=PROJECT,entity=USER,experiment=run, save_dir=os.getenv("WANDB_CACHE_DIR","."))                 #<-----CHANGE ME
        config=run.config.as_dict()

    train(config,dir,devices,accelerator,Dataset,logtool)
def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
        '#SBATCH --time={}'.format( '48:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version),
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',# Set this to GPUs per node.
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('zhangafra818@gmail.com'),                                                                                   #<-----CHANGE ME
    ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account bdlan08',
                'export CONDADIR=/nobackup/projects/bdlan08/$USER/miniconda',                                                         #<-----CHANGE ME                                                    
                'export WANDB_CACHE_DIR=/nobackup/projects/bdlan08/$USER/',
                'export MODELDIR=/nobackup/projects/bdlan08/$USER/modelckpts',
                'export TEMP=/nobackup/projects/bdlan08/$USER/',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             #add command to request more memory
                             '#SBATCH --mem=96G',
                             '#SBATCH --cpus-per-task=8',
                             'export CONDADIR=/storage/hpc/07/zhang303/conda_envs/torch',                                                     
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',
                             'export WANDB_CACHE_DIR=$global_scratch',
                             'export MODELDIR=$global_storage/modelckpts',
                             'export TEMP=$global_scratch',
                             'export ISHEC=True'])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'export WANDB_API_KEY=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        #'export ISHEC=True',                                                                                                                       
        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',
        #'pip install -r requirements.txt',                                                  
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)
    #If you're deploying prototyping code and often changing your pip env,
    # consider adding in a 'scopy requirements.txt
    # and then append command 'pip install -r requirements.txt...
    # This should add your pip file from the launch dir to the run location, then install on each node.

    sub_commands.append('srun {} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section,
    sub_commands = [x.lstrip() for x in sub_commands]

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v
if __name__ == '__main__':
    from demoparse import parser
    from subprocess import call

    myparser=parser()
    hyperparams = myparser.parse_args()

    defaultConfig=hyperparams.__dict__

    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    if NumTrials==-1:
        #debug mode - We want to just run in debug mode...
        #pick random config and have at it!

        trial=hyperparams.generate_trials(1)[0]
        #We'll grab a random trial, BUT have to launch it with KWARGS, so that DDP works.
        #result = call('{} {} --num_trials=0 {}'.format("python",os.path.realpath(sys.argv[0]),__get_hopt_params(trial)), shell=True)

        print("Running trial: {}".format(trial))

        wandbtrain(trial)

    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        wandbtrain(hyperparams)


    #LEts pretend stephen fixed something
    #OR To run with Default Args
    else:
        trials=myparser.generate_wandb_trials(entity="st7ma784",project="AllDataPGN")

        for i,trial in enumerate(trials):
            command=SlurmRun(trial)
            slurm_cmd_script_path =  os.path.join(defaultConfig.get("dir","."),"slurm_cmdtrial{}.sh".format(i))

            with open(slurm_cmd_script_path, "w") as f:
                f.write(command)
            print('\nlaunching exp...')

            
            
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
                
                #copy the file to a new folder with name time 2 days from now
                TIMETORUN=(datetime.datetime.now()+datetime.timedelta(days=3)).strftime("%Y-%m-%d")
                os.makedirs(os.path.join(defaultConfig.get("dir","."),"slurm_scripts","RunAfter{}".format(TIMETORUN)),exist_ok=True)
                os.rename(slurm_cmd_script_path,os.path.join(defaultConfig.get("dir","."),"slurm_scripts","RunAfter{}".format(TIMETORUN),"EVAL"+slurm_cmd_script_path.split("/")[-1]))

            
            else:
                print('launch failed...')
