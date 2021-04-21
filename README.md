# Reinforcement learning in Tetris
This is project that aims train an agent to play Tetris. It uses a customized gym environment that is mainly based on a pygame version of Tetris called [MaTris](https://github.com/Uglemat/MaTris).

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
