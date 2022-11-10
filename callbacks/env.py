from ..util.env import full_env_info
from ..util.ioutil import autosave


def LogEnv(experiment):

    info = full_env_info()
    autosave(info, experiment.path / "env_info.json")
