# Ghostbusters MARL 

 Multi Agent Reinforcement Learning is used to train agents to play a version of a ghostbusters game, where they must learn to surround and capture an avoiding ghost, inside of a partially observable environment.

## Main Files 

### GameEngine.py

### ghostbusters_env.py

### Agent.py 

### Ghost.py

### plot_metric.py


## Algorithms
Inside of pymarl/src/algs
* ghostbusters_qmix.yaml
* ghostbusters_vnd.yaml

## Environments
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


# Running Code 

## Initial Setup
1) Download Repo
2) Initialize virtual environments in both folders "GhostbustersMARL" and "pymarl"
  1)cd into appropriate folder
  2)run the command "python source .venv/bin/activate"
  3)run the command: "pip install -r requirements.txt"  


## How to train Models 

1) cd into pymarl folder
2) Activate venv
3) choose a learner config to use (qmix or vdn), and an environment.
4) Run the following command, with your chosen alg and environment substituted
<br><br>EXAMPLE COMMAND:    
 python src/main.py --config=ghostbusters_qmix --env-config=ghostbusters_PartObv_Stage1

#### Results
Models will be saved to: 
<br>Metrics will be saved to: 

## How to run Models 
1) cd into pymarl folder
2) Activate venv
3) Select appropriate algorithm (all saved policies use qmix)
4) Select environment you wish to use
5) Run the following command, with your chosen alg and environment substituted
<br><br>EXAMPLE COMMAND:    
python src/main.py \
  --config=ghostbusters_qmix \
  --env-config=ghostbusters_PartObv_Stage3 \
  evaluate=True



