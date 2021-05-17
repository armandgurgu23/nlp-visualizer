from backend.models.transformerModels.transformerModels import TransformerModelFactory
from configs.models.defaults.transformerDefaults import TransformerConfigHandler
import pytest


@pytest.fixture
def config_handler():
    return TransformerConfigHandler()


class TestTransformerModels:
    def test_transformer_ner_model_pipeline_initialization(self, config_handler):
        ner_pipeline_config_path = "tests/configs/transformerModels/transformerNer.yaml"
        model_initialization_config = config_handler(ner_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        assert model_object.modelPipeline is not None

    def test_transformer_sa_model_pipeline_initialization(self, config_handler):
        sa_pipeline_config_path = "tests/configs/transformerModels/transformerSA.yaml"
        model_initialization_config = config_handler(sa_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        assert model_object.modelPipeline is not None

    def test_transformer_ner_model_pipeline_prediction(self, config_handler):
        ner_pipeline_config_path = "tests/configs/transformerModels/transformerNer.yaml"
        model_initialization_config = config_handler(ner_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        # Test input for NER.
        sample_text_inputs = [
            "Apple is a large company!",
            "My name is Armand and I live in Abu Dhabi!",
        ]
        model_predictions = model_object(sample_text_inputs)
        assert 2 == 2

    def test_transformer_sa_model_pipeline_prediction(self, config_handler):
        sa_pipeline_config_path = "tests/configs/transformerModels/transformerSA.yaml"
        model_initialization_config = config_handler(sa_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        # Test input for Sentiment-Analysis.
        sample_text_inputs = ["Today is an amazing day", "I hate you"]
        model_predictions = model_object(sample_text_inputs)
        assert 2 == 2
