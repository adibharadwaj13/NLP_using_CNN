import yaml
import os


dataParams = {
    
'vocab_size':30000,
'max_review_length':80
}


modelParams = {
        
'l2_reg':1e-6,
'batch_size': 16,
'dropout': 0.5,
'lr': 0.002,
'epochs':150,
'embedding_dim': 80,
'filter_sizes':[3,4,5],
'num_filters': 30,
'num_classes':5,

'sub_batch_size': 16,
'num_sub_batches': 3200,

'valid_sub_batch_size': 16,
'valid_num_sub_batches': 800

}


globalParams = {
'build_vocab':False,
'convert_and_pad':False
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