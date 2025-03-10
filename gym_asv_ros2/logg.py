from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import re

import stable_baselines3.common.logger as sb3_logger
from pathlib import Path


class FileStorage:
    def __init__(self, root_dir: str="training", id: str="") -> None:

        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            raise FileExistsError(f"{self.work_dir} does not exists. Please create it.")

        self.work_dir = self.root_dir / id

        self.episode_summary = self.work_dir / "episode_summary"
        self.info = self.work_dir / "info"
        self.tesnserflow = self.work_dir / "tesnserflow"
        self.agents = self.work_dir / "agents"
        self.videos = self.work_dir / "video"


    def init_storage(self):
        """Create the file structre if it does not exists"""

        for attr_name, attr in self.__dict__.items():
            if attr_name == "work_dir":
                continue
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True, parents=True)


    def verify_filestorage_choise(self) -> bool:
        """Returns true if we should continue, False if not."""

        if not self.work_dir.exists():
            return True

        option = input(f"{self.work_dir} Does allready exists, are you sure you want to use this logdir? [y/N]")
        if option.lower() == "y":
            self.info = self.info
            return True

        
        return False

    def agent_picker(self, name="") -> str:

        agents = [file.name for file in self.agents.iterdir() if file.is_file()]

        if len(agents) <= 0:
            print("There is no agents in file storage")
            return ""

        if name in agents:
            return str( self.agents / name )

        test = agents[0]
        reg = re.match(r"(\d+)__", test).group(1)
        print(f"expression: {test} gives {reg}")

        def sort_func(x):
            m = re.match(r"(\d+)__", x)
            return int(m.group(1)) if m else -1

        agents.sort(key=sort_func)

        # Prompt for agent to use
        for i in range(len(agents)):
            print(i, agents[i])

        try:
            index = int(input("Which agent [int]? "))
            return str( self.agents / agents[index] )

        except ValueError:
            print("Invalid Choise.")
        
        return ""


class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, episode_log_dir: str, agents_dir: str, verbose: int = 0, save_frequency: int = 10000):
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
        self.save_frequency = save_frequency

        # history variables
        self.info_history = {"total_episodes": 0, "successful_episodes": 0}

        # Configure episode logger used to log the summary of each episode
        self.agents_dir = agents_dir
        self.episode_logger = sb3_logger.configure(episode_log_dir, ["csv", "tensorboard"]) # NOTE: Will write over if log file allready exists

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # self.save_frequency = int(self.model._total_timesteps / 10) # FIXME: Is wrong when loading existing model
        print(f"[TrainingCallback]: self.num_timesteps: {self.num_timesteps}, model.num_timesteps: {self.model.num_timesteps}")

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
        ids = np.where(done_array)[0]
        if np.any(done_array):
            infos = np.array(self.locals.get("infos"))[done_array]
            episode_summaries = np.array(self.training_env.get_attr("episode_summary"))[done_array]

            for i in range(len(infos)):
                log_statistics = episode_summaries[i]
                desired_from_info = ["TimeLimit.truncated", "episode"]
                log_statistics.update({k: infos[i][k] for k in desired_from_info if k in infos[i]}) # Copy the desired metrics from infos

                # Update general metrics
                self.info_history["successful_episodes"] += int(log_statistics["reached_goal"])
                self.info_history["total_episodes"] += 1
                self.logger.record_mean("info/epsiode_reward", log_statistics["episode"]["r"]) # log the reward on episode termination

                record_nested_dict(
                    self.episode_logger.record, log_statistics
                )
                self.episode_logger.record("env_id", ids[i])
                self.episode_logger.dump(self.num_timesteps)

        # if self.num_timesteps % self.log_frequency == 0:
        #     pass

        # NOTE: Consider saving agent before policy update instead on a fixed frequency
        if self.num_timesteps % self.save_frequency == 0:
            filename = f"{self.agents_dir}/{self.num_timesteps}__{self.info_history['total_episodes']}"
            self.logger.info(f"Saving agent at {filename}")
            self.model.save(filename)

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

        # Recursive call to nest out the dict
        if isinstance(value, dict):
            record_nested_dict(out, value, current_key)

        elif isinstance(value, np.ndarray):
            flat_values = value.flatten()
            for i in range(len(flat_values)):
                out(f"{current_key}/{i}", flat_values[i])

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
    test = FileStorage("training", "lidar_with_dock_obst_5KK")
    test.agent_picker()
    # print(f"Storage was: {test.work_dir}")
    # test = test.new_sub_storage_if_exists()
    # print(f"Storage is: {test.work_dir}")
    
    # f = test.agent_picker(name="200000__133.zip")
    # print(f)

    
    # print(test.work_dir)
    # test.init_storage()
