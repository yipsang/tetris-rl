# Based on https://github.com/ikostrikov/pytorch-a3c

import logging
import torch.multiprocessing as mp
import gym

from a3c_adam import A3CAdamOpt
from a3c_test import test
from a3c_train import train
from actor_critic_value_network import ActorCriticValueNetwork
from wrappers import PositionAction

logging.basicConfig(level=logging.INFO)

def main():
    num_processes = 1

    env = gym.make("matris-v0", render=False, timestep=0.05)
    env = PositionAction(env, handcrafted_features=True)

    shared_model = ActorCriticValueNetwork(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()

    optimizer = A3CAdamOpt(shared_model.parameters())
    optimizer.share_memory()
    
    processes = []

    args = {}

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(num_processes, model))
    # p.start()
    # processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()