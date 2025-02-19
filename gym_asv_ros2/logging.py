from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, logdir, verbose: int = 0):
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
        
        self.logdir = logdir
        self.episodes = 0
        self.log_frequency = 100

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
        episodes_finished = np.sum(done_array)
        if episodes_finished > 0:
            self.logger.info(f"{episodes_finished} episodes is finished")
            # self.episodes += np.sum(done_array)
            # self.logger.record("test/episodes", self.episodes)

            infos = np.array(self.locals.get("infos"))[done_array]
            for i in range(len(infos)):
                for key, value in infos[i].items():
                    self.logger.record(f"info/env_{i}/{key}", value=value)


        # Log every <log_frequency> iteration
        if self.num_timesteps % self.log_frequency == 0:
            # self.logger.record("", self.log_frequency)
            # self.logger.info(self.locals.get("infos"))
            pass


        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.info("updating policy:")
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
