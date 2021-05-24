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
        return self.processPredictions(rawPredictions, inputText)

    def processPredictions(self, predictions, inputText):
        if self.modelConfig.pipeline.pipeline_type == "ner":
            return self.processNerPredictions(predictions, inputText)
        elif self.modelConfig.pipeline.pipeline_type == "sentiment-analysis":
            return self.processSaPredictions(predictions, inputText)
        else:
            raise NotImplementedError(
                "Requested model pipeline currently not supported!"
            )

    def getModelPredictions(self, inputText):
        return self.modelPipeline(inputText)

    def processNerPredictions(self, predictions, inputText):
        return self.mergeOverlappingEntityTypes(predictions, inputText)

    def mergeOverlappingEntityTypes(self, nerPredictions, inputText):
        # Since some entity types may have multiple tokens (ie: Abu Dhabi) we want
        # to merge multi-token identical entity types to one prediction.
        outputEntities = []
        for batchIndex, currBatchElementPrediction in enumerate(nerPredictions):
            batchText = inputText[batchIndex]
            previousEntity = None
            processedNerEntities = []
            indexesToIgnore = None
            for predIndex, currRawPrediction in enumerate(currBatchElementPrediction):
                if indexesToIgnore and predIndex in indexesToIgnore:
                    continue
                if previousEntity and previousEntity.get(
                    "entity"
                ) == currRawPrediction.get("entity"):
                    (
                        matchingTerminatingBlock,
                        indexesToIgnore,
                        confidenceScores,
                    ) = self.findTerminatingEntityBlock(
                        currBatchElementPrediction,
                        predIndex,
                        currRawPrediction.get("entity"),
                    )
                    # Should also remove the previous block when
                    # constructing the merged block.
                    processedNerEntities.pop()
                    finalText, entityType = self.mergePreviousAndTerminatingBlocks(
                        matchingTerminatingBlock, previousEntity, batchText
                    )
                    processedNerEntities.append(
                        {"entityText": finalText, "entityLabel": entityType}
                    )
                else:
                    processedPrediction = {
                        "entityText": self.extractEntityTextFromInput(
                            batchText,
                            currRawPrediction.get("start"),
                            currRawPrediction.get("end"),
                        ),
                        "entityLabel": currRawPrediction.get("entity"),
                    }
                    processedNerEntities.append(processedPrediction)
                previousEntity = currRawPrediction
            outputEntities.append(processedNerEntities)
        return outputEntities

    def mergePreviousAndTerminatingBlocks(
        self, terminatingBlock, previousBlock, inputText
    ):
        startPrevious = previousBlock.get("start")
        endTerminating = terminatingBlock.get("end")
        # We want to slice the original text at the start and end indexes in order to get the full
        # input text.
        entityText = self.extractEntityTextFromInput(
            inputText, startPrevious, endTerminating
        )
        return entityText, terminatingBlock.get("entity")

    def extractEntityTextFromInput(self, inputText, start, end):
        return inputText[start:end]

    def findTerminatingEntityBlock(self, nerPredictions, predIndex, currEntityType):
        terminatingEntityBlock = None
        indexesToIgnore = []
        confidenceScores = []
        for currEndIndex in range(predIndex, len(nerPredictions)):
            currEntityBlock = nerPredictions[currEndIndex]
            if currEntityBlock.get("entity") == currEntityType:
                terminatingEntityBlock = currEntityBlock
                indexesToIgnore.append(currEndIndex)
                confidenceScores.append(currEntityBlock.get("score"))
        return terminatingEntityBlock, indexesToIgnore, confidenceScores

    def processSaPredictions(self, predictions, inputText):
        # For now there's no important post-processing to apply
        # to the sentiment predicted.
        return predictions

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
