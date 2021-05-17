from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)


class TransformerModelFactory(object):
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        self.modelRegistry = {
            "ner": AutoModelForTokenClassification,
            "sentiment-analysis": AutoModelForSequenceClassification,
        }
        self.model = self.initializeModel(self.modelConfig)
        # Initialize the full model pipeline, which involves
        # tokenization.
        self.modelPipeline = self.initializeModelPipeline(self.modelConfig, self.model)
        print(self.modelPipeline)
        print("Finished initializing model pipeline above!")

    def __call__(self, inputText):
        rawPredictions = self.getModelPredictions(inputText)
        return self.processPredictions(rawPredictions)

    def processPredictions(self, predictions):
        if self.modelConfig.pipeline.pipeline_type == "ner":
            return self.processNerPredictions(predictions)
        elif self.modelConfig.pipeline.pipeline_type == "sentiment-analysis":
            return self.processSaPredictions(predictions)
        else:
            raise NotImplementedError(
                "Requested model pipeline currently not supported!"
            )

    def getModelPredictions(self, inputText):
        return self.modelPipeline(inputText)

    def processNerPredictions(self, predictions):
        print(predictions)
        print("THESE ARE THE RAW NER PREDICTIONS!")
        raise NotImplementedError("Set up code to post-process NER predictions!")

    def processSaPredictions(self, predictions):
        print(predictions)
        print("THESE ARE THE RAW SA PREDICTIONS!")
        raise NotImplementedError("Set up code to post-process SA predictions!")

    def initializeModel(self, config):
        modelToInitialize = self.modelRegistry[config.pipeline.pipeline_type]
        # Load a specific checkpoint associated to that model.
        initializedModel = modelToInitialize.from_pretrained(
            config.pipeline.pipeline_model
        )
        initializedModel.eval()
        return initializedModel

    def initializeModelPipeline(self, config, model):
        tokenizer = self.initializeTokenizer(config)
        return pipeline(config.pipeline.pipeline_type, model=model, tokenizer=tokenizer)

    def initializeTokenizer(self, config):
        return AutoTokenizer.from_pretrained(config.pipeline.pipeline_model)
