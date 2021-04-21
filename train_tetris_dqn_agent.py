import logging

from dqn_agent_trainer import TetrisDQNAgentTrainer

logging.basicConfig(level=logging.INFO)


def main():
    trainer = TetrisDQNAgentTrainer(
        layers_size=[128, 256],
        buffer_size=90000,
        batch_size=256,
        gamma=1,
        lr=1e-4,
        freeze_step=1,
        gpu=False,
        episodes=10000,
        render=False,
        random_action_episodes=100,
        episolon_decay_episodes=5000,
        start_eps=1,
        end_eps=1e-4,
        max_episode_step=1000,
        handcrafted_features=True,
        conv_layers_config=[(2, 1, 0, 32), (2, 2, 1, 32), (1, 1, 0, 32)],
        conv_linear_size=256,
        is_noisy=True,
    )
    losess, rewards = trainer.start()


if __name__ == "__main__":
    main()