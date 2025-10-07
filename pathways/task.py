# Ignore deprecation warnings from gymnasium
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

import mod_cog
import neurogym as ngym
import torch
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers import ScheduleEnvs
from torch.utils.data import DataLoader

from .dataset import get_dataset


class Task:
    """
    A task on which to train or evaluate a model. A task may consist of one or many
    environments, each of which can be a task from the modcog or neurogym task suites.
    """

    name: str
    input_size: int
    output_size: int

    @classmethod
    def from_id(cls, task_id: str) -> "Task":
        """
        Create a task from a task id.

        :param: task_id: The id of the task. 'modcog/all' selects all 82 modcog tasks,
            'modcog/16' selects a group of 16 modcog tasks, and 'modcog/<env_id>'
            selects a single modcog task.
        """
        if task_id.startswith("modcog/"):
            env_id = task_id.split("/")[1]
            if env_id == "all":
                return cls.all_modcog()
            if env_id == "16":
                return cls.modcog_16()
            return cls.from_modcog(env_id)
        if task_id.startswith("neurogym/"):
            return cls.from_neurogym(task_id.split("/")[1])

        msg = f"Unknown task: {task_id}"
        raise ValueError(msg)

    @classmethod
    def all_modcog(cls) -> "Task":
        """Create a task using all 82 tasks from modcog."""

        env_names = [
            "go",
            "rtgo",
            "dlygo",
            "anti",
            "rtanti",
            "dlyanti",
            "dm1",
            "dm2",
            "ctxdm1",
            "ctxdm2",
            "multidm",
            "dlydm1",
            "dlydm2",
            "ctxdlydm1",
            "ctxdlydm2",
            "multidlydm",
            "dms",
            "dnms",
            "dmc",
            "dnmc",
            "dlygointr",
            "dlygointl",
            "dlyantiintr",
            "dlyantiintl",
            "dlydm1intr",
            "dlydm1intl",
            "dlydm2intr",
            "dlydm2intl",
            "ctxdlydm1intr",
            "ctxdlydm1intl",
            "ctxdlydm2intr",
            "ctxdlydm2intl",
            "multidlydmintr",
            "multidlydmintl",
            "dmsintr",
            "dmsintl",
            "dnmsintr",
            "dnmsintl",
            "dmcintr",
            "dmcintl",
            "dnmcintr",
            "dnmcintl",
            "goseqr",
            "rtgoseqr",
            "dlygoseqr",
            "antiseqr",
            "rtantiseqr",
            "dlyantiseqr",
            "dm1seqr",
            "dm2seqr",
            "ctxdm1seqr",
            "ctxdm2seqr",
            "multidmseqr",
            "dlydm1seqr",
            "dlydm2seqr",
            "ctxdlydm1seqr",
            "ctxdlydm2seqr",
            "multidlydmseqr",
            "dmsseqr",
            "dnmsseqr",
            "dmcseqr",
            "dnmcseqr",
            "goseql",
            "rtgoseql",
            "dlygoseql",
            "antiseql",
            "rtantiseql",
            "dlyantiseql",
            "dm1seql",
            "dm2seql",
            "ctxdm1seql",
            "ctxdm2seql",
            "multidmseql",
            "dlydm1seql",
            "dlydm2seql",
            "ctxdlydm1seql",
            "ctxdlydm2seql",
            "multidlydmseql",
            "dmsseql",
            "dnmsseql",
            "dmcseql",
            "dnmcseql",
        ]

        envs = [getattr(mod_cog, env_name)() for env_name in env_names]
        schedule = RandomSchedule(len(envs))

        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        dataset = ngym.Dataset(env, batch_size=64, seq_len=350)
        self = cls("modcog/all", env.observation_space.shape[0], env.action_space.n)
        self.dataset = dataset
        self.schedule = schedule
        self.env = env
        self.env_names = env_names
        return self

    @classmethod
    def modcog_16(cls) -> "Task":
        """Create a task using a selection of 16 tasks from modcog."""

        env_names = [
            "go",
            "rtgo",
            "dlygo",
            "anti",
            "rtanti",
            "dlyanti",
            "ctxdlydm1intr",
            "dm2",
            "ctxdm1",
            "dnmcintl",
            "antiseql",
            "dlydm1",
            "dlydm2",
            "dnmsseql",
            "ctxdlydm2",
            "multidlydm",
        ]

        envs = [getattr(mod_cog, env_name)() for env_name in env_names]
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        dataset = ngym.Dataset(env, batch_size=64, seq_len=350)
        self = cls("modcog/16", env.observation_space.shape[0], env.action_space.n)
        self.dataset = dataset
        self.env = env
        self.env_names = env_names
        return self

    @classmethod
    def from_modcog(cls, env_id: str) -> "Task":
        """
        Create a task using an environment from the modcog task suite.

        :param: env_id: Name of the modcog task.
        """

        env = getattr(mod_cog, env_id)()
        dataset = ngym.Dataset(
            env,
            batch_size=64,
            seq_len=350,
        )
        self = cls(
            f"modcog/{env_id}",
            env.observation_space.shape[0],
            env.action_space.n,
        )
        self.dataset = dataset
        self.env = env
        self.env_names = [env_id]
        return self

    @classmethod
    def from_neurogym(cls, env_id: str) -> "Task":
        """
        Create a task using an environment from the neurogym task suite.

        :param: env_id: Name of the neurogym task.
        """

        dataset = ngym.Dataset(
            env_id,
            env_kwargs={"dt": 100},
            batch_size=64,
            seq_len=100,
        )
        env = dataset.env
        self = cls(
            f"neurogym/{env_id}",
            env.observation_space.shape[0],
            env.action_space.n,
        )
        self.dataset = dataset
        self.env = env
        return self

    def __init__(self, name: str, input_size: int, output_size: int) -> None:
        """
        Initialize the task.

        :param: name: The name of the task.
        :param: input_size: The size of the input.
        :param: output_size: The size of the output.
        """

        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.dataset = None

    def get_dataloader(
        self,
        *,
        num_samples: int,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 8,
    ) -> DataLoader:
        """
        Get a dataloader for the task.

        :param: num_samples: Number of samples to get from the dataset.
        :param: batch_size: Batch size.
        :param: shuffle: Whether to shuffle the dataset.
        :param: num_workers: Number of workers to use for the dataloader.
        :return: A dataloader for the task.
        """

        name = self.name

        dataset = get_dataset(name, self.name, num_samples, self.dataset)
        dataloader_kwargs = {}

        if torch.cuda.is_available():
            dataloader_kwargs["pin_memory"] = True
            dataloader_kwargs["pin_memory_device"] = "cuda"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            **dataloader_kwargs,
        )
