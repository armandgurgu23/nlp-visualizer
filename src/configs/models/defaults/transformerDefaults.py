# my_project/config.py
from yacs.config import CfgNode as CN


class TransformerConfigHandler(object):
    def __init__(self):
        self.initializeYTConfig()

    def __call__(self, config_filepath):
        return self.getTransformersConfig(config_filepath)

    def getTransformersConfig(self, config_filepath):
        defaults_config = self._transformers.clone()
        defaults_config.merge_from_file(config_filepath)
        defaults_config.freeze()
        return defaults_config

    def initializeYTConfig(self):
        self._transformers = CN()
        self._transformers.pipeline = CN()
        self._transformers.pipeline.pipeline_type = "sentiment-analysis"
        self._transformers.pipeline.pipeline_model = ""
        return
