from backend.models.transformerModels.transformerModels import TransformerModelFactory
from backend.configs.models.defaults.transformerDefaults import TransformerConfigHandler
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
            "My name is Bob and I live in Abu Dhabi!",
            "My name is Bob Alex Martinez and I live in Toronto Ontario Canada!",
        ]
        model_predictions = model_object(sample_text_inputs)
        assert model_predictions[-1][-1].get("entityText") == "Toronto Ontario Canada"

    def test_transformer_ner_model_pipeline_process_predictions(self, config_handler):
        ner_pipeline_config_path = "tests/configs/transformerModels/transformerNer.yaml"
        model_initialization_config = config_handler(ner_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        # Test input for the NER post-processing method.
        sample_text_inputs = [
            "Apple is a large company!",
            "My name is Bob and I live in Abu Dhabi!",
            "My name is Bob Alex Martinez and I live at 456 Newhaven London!",
        ]
        sample_ner_raw_predictions = [
            [
                {
                    "word": "apple",
                    "score": 0.9965875148773193,
                    "entity": "I-ORG",
                    "index": 1,
                    "start": 0,
                    "end": 5,
                }
            ],
            [
                {
                    "word": "bob",
                    "score": 0.9977469444274902,
                    "entity": "I-PER",
                    "index": 4,
                    "start": 11,
                    "end": 14,
                },
                {
                    "word": "abu",
                    "score": 0.9996529221534729,
                    "entity": "I-LOC",
                    "index": 9,
                    "start": 29,
                    "end": 32,
                },
                {
                    "word": "dhabi",
                    "score": 0.9987406730651855,
                    "entity": "I-LOC",
                    "index": 10,
                    "start": 33,
                    "end": 38,
                },
            ],
            [
                {
                    "word": "bob",
                    "score": 0.9975388050079346,
                    "entity": "I-PER",
                    "index": 4,
                    "start": 11,
                    "end": 14,
                },
                {
                    "word": "alex",
                    "score": 0.9991857409477234,
                    "entity": "I-PER",
                    "index": 5,
                    "start": 15,
                    "end": 19,
                },
                {
                    "word": "martinez",
                    "score": 0.9938157796859741,
                    "entity": "I-PER",
                    "index": 6,
                    "start": 20,
                    "end": 28,
                },
                {
                    "word": "new",
                    "score": 0.9955864548683167,
                    "entity": "I-LOC",
                    "index": 13,
                    "start": 47,
                    "end": 50,
                },
                {
                    "word": "##haven",
                    "score": 0.9858444929122925,
                    "entity": "I-LOC",
                    "index": 14,
                    "start": 50,
                    "end": 55,
                },
                {
                    "word": "london",
                    "score": 0.982772171497345,
                    "entity": "I-LOC",
                    "index": 15,
                    "start": 56,
                    "end": 62,
                },
            ],
        ]
        model_predictions = model_object.processNerPredictions(
            sample_ner_raw_predictions, sample_text_inputs
        )
        assert (
            model_predictions[-1][0].get("entityText") == "Bob Alex Martinez"
            and model_predictions[-1][0].get("entityLabel") == "I-PER"
        )

    def test_transformer_sa_model_pipeline_prediction(self, config_handler):
        sa_pipeline_config_path = "tests/configs/transformerModels/transformerSA.yaml"
        model_initialization_config = config_handler(sa_pipeline_config_path)
        model_object = TransformerModelFactory(model_initialization_config)
        # Test input for Sentiment-Analysis.
        sample_text_inputs = ["Today is an amazing day", "I hate you"]
        model_predictions = model_object(sample_text_inputs)
        assert (
            model_predictions[0].get("label") == "happy"
            and model_predictions[-1].get("label") == "angry"
        )
