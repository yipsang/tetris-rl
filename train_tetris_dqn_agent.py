import logging

from dqn_agent_trainer import TetrisDQNAgentTrainer

logging.basicConfig(level=logging.INFO)


def main():
    trainer = TetrisDQNAgentTrainer(
        layers_size=[128, 256],
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        freeze_step=5,
        gpu=False,
        episodes=10000,
        render=False,
        random_action_episodes=100,
        episolon_decay_episodes=750,
        start_eps=1,
        end_eps=0.01,
        max_episode_step=1000,
        save_every=10,
    )
    losess, rewards = trainer.start()


if __name__ == "__main__":
    main()