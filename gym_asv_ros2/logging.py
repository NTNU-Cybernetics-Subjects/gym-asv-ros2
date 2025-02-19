from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import stable_baselines3.common.logger as sb3_logger
from pathlib import Path


class FileStorage:
    def __init__(self, work_dir: str="training") -> None:
        self.work_dir = Path(work_dir).resolve()
        if not self.work_dir.exists():
            raise FileExistsError(f"{self.work_dir} does not exists. Please create it.")

        self.episode_summary = self.work_dir.joinpath("episode_summary")
        self.info = self.work_dir.joinpath("info")
        self.tesnserflow = self.work_dir.joinpath("tesnserflow")
        self.agents = self.work_dir.joinpath("agents")


    def init_storage(self):
        """Create the file structre if it does not exists"""

        for attr_name, attr in self.__dict__.items():
            if attr_name == "work_dir":
                continue
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True, parents=True)




class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, episode_log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.episodes = 0
        self.log_frequency = 100

        # history variables
        self.info_history = {"total_episodes": 0, "successful_episodes": 0}

        # Configure episode logger used to log the summary of each episode
        self.episode_logger = sb3_logger.configure(episode_log_dir, ["csv", "stdout"])

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        # Check if any episodes are done
        done_array = np.array(self.locals.get("dones"))
        # episodes_finished = np.sum(done_array)
        if np.any(done_array):
            infos = np.array(self.locals.get("infos"))[done_array]
            for i in range(len(infos)):
                self.info_history["successful_episodes"] += int(infos[i]["step_info"]["reached_goal"])
                self.info_history["total_episodes"] += 1

                record_nested_dict(
                    self.episode_logger.record, infos[i], prefix=f"env_{i}"
                )
                # record_nested_dict(self.logger.record_mean, infos[i]) # Record the mean of all the episodes that have finished since last dump
            self.episode_logger.dump()


        # if self.num_timesteps % self.log_frequency == 0:
        #     pass

        return True

    # NOTE: self.logger seems to also dump after this function call
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.info("updating policy:")
        record_nested_dict(self.logger.record, self.info_history, "info")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def record_nested_dict(out, info, prefix=""):
    """Record a nested dictionary to out, out must be a callable function that
    takes in key, value"""

    for key, value in info.items():
        current_key = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            record_nested_dict(out, value, current_key)
        else:
            out(current_key, value)


def test_record_nested_dict():
    test_dict = {
        "first": 10,
        "second": 50,
        "third": {"first": 2, "second": {"first": 8, "second": 10}, "third": 8},
    }
    t = []

    def out(key, value):
        t.append(f"{key}: {value}")

    record_nested_dict(out, test_dict, prefix="test")
    # print("finsihed:\n", t)


if __name__ == "__main__":
    # test_record_nested_dict()
    test = FileStorage("testtt")
    test.init_storage()
