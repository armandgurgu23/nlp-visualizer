import spacy

class SpacyWrapper(object):
    def __init__(self, modelToLoad = 'en_core_web_md'):
        self.modelToLoad = modelToLoad
        self.spacyModel = self.initializeSpacyModel(self.modelToLoad)
    
    def __call__(self, inputText):
        return self.getFullPredictionTree(inputText)

    def initializeSpacyModel(self, modelToLoad):
        return spacy.load(modelToLoad)
    
    def getFullPredictionTree(self, inputText):
        predictions = self.runSpacyModelInference(inputText)
        return predictions.to_json()

    def runSpacyModelInference(self, inputText):
        return self.spacyModel(inputText)