
from backend.models.spacyModels import spacyModel


class TestSpacyModel:
    def test_spacy_wrapper_model_inference_case_1(self):
        natural_language_utterance = "Apple is a large company!"
        model_predictions = spacyModel(natural_language_utterance)
        detected_entities = spacyModel.extractPredictedEntities(model_predictions, natural_language_utterance)
        assert detected_entities[0]['entityText'] == "Apple" and detected_entities[0]['entityLabel'] == 'ORG'

