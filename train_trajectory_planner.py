import logging

from trajectory_planner_trainer import TrajectorPlannerTrainer

logging.basicConfig(level=logging.INFO)


def main():
    trainer = TrajectorPlannerTrainer()
    losess, rewards = trainer.start()


if __name__ == "__main__":
    main()