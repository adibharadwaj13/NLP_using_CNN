import yaml
import os


dataParams = {
    
'vocab_size':30000,
'max_review_length':80
}


modelParams = {
    
'l2_reg':1e-5,
'batch_size': 8,
'dropout': 0.3,
'lr': 0.004,
'epochs':25,
'embedding_dim': 50,
'filter_sizes':[2,3,4],
'num_filters': 20,
'num_classes':5,
'num_valid_samples':5000
}


globalParams = {
'build_vocab':True,
'convert_and_pad':True
}




def loadfeatureParams(filepath):
    ''' returns: Feature Parmas Dict'''

    stream = open(filepath, 'r')
    return yaml.load(stream)


def DumpfeatureParams(Paramdict, filename):
    ''' returns: Path to FeatureParmas.yaml'''

    

    filepath = os.path.join(os.getcwd(), filename)
    stream = open(filepath, 'w')
    yaml.dump(Paramdict, stream)
    return filepath



def createParamsFile():
  yamlfile = DumpfeatureParams(modelParams,'modelParams.yaml')

  file1 = 'dataParams.yaml'
  yamlfile = DumpfeatureParams(dataParams,file1)

  file1 = 'globalParams.yaml'
  yamlfile = DumpfeatureParams(globalParams,file1)