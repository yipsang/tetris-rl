# Based on https://github.com/ikostrikov/pytorch-a3c

import logging
import torch.multiprocessing as mp
import gym

from adqn_adam import A3CAdamOpt, SharedAdam
from adqn_test import test
from adqn_train import train
from value_network import ValueNetwork
from wrappers import PositionAction

logging.basicConfig(level=logging.INFO)

def main():
    num_processes = 8

    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(env, handcrafted_features=True)

    shared_model = ValueNetwork(env.observation_space.shape[0])
    shared_model.share_memory()

    shared_model_target = ValueNetwork(env.observation_space.shape[0])
    shared_model_target.share_memory()

    optimizer = SharedAdam(shared_model.parameters())
    optimizer.share_memory()
    
    processes = []

    args = {}

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(num_processes, model))
    # p.start()
    # processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, shared_model_target, counter, lock, optimizer))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()