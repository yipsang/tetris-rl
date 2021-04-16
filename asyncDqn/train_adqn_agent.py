# Based on https://github.com/ikostrikov/pytorch-a3c

import platform
import logging
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from torch.utils.tensorboard.writer import SummaryWriter
import gym

from adqn_adam import SharedAdam
from adqn_test import test
from adqn_train import train
from value_network import ValueNetwork
from wrappers import PositionAction

logging.basicConfig(level=logging.INFO)

class SummaryWriterProxy(object):
    def __init__(self):
        self.writer = SummaryWriter()
        self.counter = 0

    def log(self, stat):
        reward, episode_len = stat
        self.writer.add_scalar("Number of steps/train", episode_len, self.counter)
        self.writer.add_scalar("Mean reward/train", reward/episode_len, self.counter)
        self.writer.add_scalar("Last episode rewards/train", reward, self.counter)
        self.counter += 1

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

    BaseManager.register('SummaryWriterProxy', SummaryWriterProxy)
    manager = BaseManager()
    manager.start()
    writer_proxy = manager.SummaryWriterProxy()

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, shared_model_target, counter, lock, optimizer, writer_proxy))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    # For MacOS
    if platform.system() == 'Darwin':
        mp.set_start_method("spawn")
    main()