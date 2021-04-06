import spacy

class SpacyWrapper(object):
    def __init__(self, modelToLoad = 'en_core_web_md'):
        self.modelToLoad = modelToLoad
        self.spacyModel = self.initializeSpacyModel(self.modelToLoad)
        print('Finished initializing the spacy model = {}'.format(modelToLoad))
    
    def __call__(self, inputText):
        return self.getFullPredictionTree(inputText)

    def initializeSpacyModel(self, modelToLoad):
        return spacy.load(modelToLoad)
    
    def getFullPredictionTree(self, inputText):
        predictions = self.runSpacyModelInference(inputText)
        return predictions.to_json()

    def runSpacyModelInference(self, inputText):
        return self.spacyModel(inputText)
    
    def extractPredictedEntities(self, modelPredictions, utteranceText):
        predEntities = modelPredictions.get('ents')
        if len(predEntities) >= 1:     
            return self.processEntities(predEntities, utteranceText)
        else:
            raise RuntimeError('Could not detect NER tokens in the utteranceText!')

    def processEntities(self,predEntities, utteranceText):
        finalEntities = []
        for currEntity in predEntities:
            startIndex = currEntity['start']
            endIndex = currEntity['end']
            entityLabel = currEntity['label']
            entityObject = {'entityText': utteranceText[startIndex:endIndex], 'entityLabel':entityLabel}
            finalEntities.append(entityObject)
        return finalEntities



