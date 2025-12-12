import os
import shutil


class AttrDict(dict):
    """"
    Custom dictionary subclass that allows accessing its keys as if they were attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    """"
    To create a .json file of the current model configurations, if the path is not exist.
    config: where .json file of the model's configs, it is passed in the command line.
    config_name: config.json
    path: the path of the checkpoint
    """
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
