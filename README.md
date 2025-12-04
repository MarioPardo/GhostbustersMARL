# Ghostbusters MARL 

 Multi Agent Reinforcement Learning is used to train agents to play a version of a ghostbusters game, where they must learn to surround and capture an avoiding ghost, inside of a partially observable environment.

![Image](https://github.com/user-attachments/assets/1f803eef-6aab-4d7f-9cbd-7ebd4b08f613)

##  --- Main Files 

### GameEngine.py
Contains game logic and "runs" everything together. Contains step, rewards, observations, state. This is where majority of work of the project was done

### ghostbusters_env.py
Wrapper to control GameEngine with PyMARL

### Agent.py 
Basic agent logic

### Ghost.py
Ghost logic and behaviour

## Grid.py
Holds important information about grid and is responsible for rendering

### plot_metric.py
Used to generate useful graphics from metric output when training and running models


## --- Algorithms
Inside of pymarl/src/algs
* ghostbusters_qmix.yaml
* ghostbusters_vnd.yaml

## --- Environments
Inside of pymarl/src/config/envs

### Full Obs
* ghostbusters_FullObv_CatchStage1.yaml : Environment used for full observability, ghost not moving
* ghostbusters_FullObv_CatchStage2.yaml : Environment used for full observability, ghost fully moving
* ghostbusters_FullObv_CatchNExtract.yaml : Environment used for full observability, catching AND extracting (this failed) 
### Partial Obs
* ghostbusers_PartObv_Stage1.yaml : Environment used for partial observability, no ghost movement
* ghostbusers_PartObv_Stage2.yaml : Environment used for partial observability, some ghost movement, smaller radius
* ghostbusers_PartObv_Stage3.yaml : Environment used for partial observability, full ghost movement, even smaller radius

## Saved Policies
Inside of pymarl/results/models

All appropriately named to their status and what environment they were run with. 
FullObv Catch winner can be used with FullObv Stage 1, 2, but was evaluated with stage 2


# --- Running Code 

## Initial Setup
1) Download Repo
2) Initialize virtual environments in both folders "GhostbustersMARL" and "pymarl"
<br>&nbsp;&nbsp; 1)cd into appropriate folder
<br>&nbsp;&nbsp; 2)run the command "python source .venv/bin/activate"
<br>&nbsp;&nbsp; 3)run the command: "pip install -r requirements.txt"  


## How to train Models 

1) cd into pymarl folder
2) Activate venv
3) choose a learner config to use (qmix or vdn), and an environment.
4) Run the following command, with your chosen alg and environment substituted
<br><br>EXAMPLE COMMAND:    
 python src/main.py --config=ghostbusters_qmix --env-config=ghostbusters_PartObv_Stage1

#### Results
Models will be saved to:  pymarl/results/models
<br>Metrics will be saved to:  pymarl/results/sacred
    <br> under the name of the environment which you selected

## How to run Models 
1) cd into pymarl folder
2) Activate venv
3) Select appropriate algorithm (all currently saved policies use qmix)
4) Select environment you wish to use
5) Run the following command, with your chosen alg and environment substituted
<br><br>EXAMPLE COMMAND:    
python src/main.py \
  --config=ghostbusters_qmix \
  --env-config=ghostbusters_PartObv_Stage3 \\
  <br>with evaluate=True \\
  <br>render=True \\
  <br>checkpoint_path="results/models/QMIX_PartObv_Stage3_Winner"
 

## How to create Graphs
1) cd into main GhostbustersMARL folder
2) init venv
3) check plot_metric.py, and update the variable METRICS_PATH_P1 to reflect the name of the folder of the results you wish to graph
4) run command "python plot_metric.py"
5) When prompted, enter the name of the "run" you wish to graph. This is a subfolder of the folder which you entered, the largest number is the greatest run
6) Enter the name of the saved plot you wish to make. It will be saved under the root folder. 
