# Reinforcement learning in Tetris
This is a project that aims train an agent to play Tetris using reinforcement learning. It uses a customized gym environment that is mainly based on a pygame version of Tetris called [MaTris](https://github.com/Uglemat/MaTris).

## Install dependencies
```
pip install -r requirements.txt
```

## Run value network training
```
python train_tetris_dqn_agent.py 
```

## Demo trained model
```
python evaluate_dqn_agent.py
```

## Validate trained model
```
python validate_dqn_agent.py
```

## Run async value network training
```
cd asyncDqn
python train_adqn_agent.py
```
