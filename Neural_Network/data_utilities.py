import h5py,yaml
import sys,os
import numpy as np 
import deepdish as dd 
from Params import loadfeatureParams,createParamsFile,DumpfeatureParams
from collections import defaultdict
import collections,re
#not currently using this

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab_and_dump(dataParams, source_file):

    vocab_size = dataParams['vocab_size'] -1 #increases by 1 when using padding token

    #first run through training set and index all words 
    words = []

    idx = 0
    #get all the words first into a giant list:
   
    with open(source_file) as f:
        for l in f:
            

            
            

            try:
                l = eval(l)
                
                filtered_text = l['Text']

                curr_review_words = clean_str(filtered_text)
                idx+=1    
                print(idx)
            
                words.extend( list (map(  str.strip, curr_review_words.split() )) )
            except:
                continue
            


    #index unknown token(last-1)



    count = [('UNK', -1)]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = defaultdict(int)
    for word, _ in count:
        dictionary[word] = len(dictionary)

    filename = 'words_and_indices.yaml'

    print("current size of vocabulary: ", len(dictionary))
    print("wrote word2int dict to %s"  %filename)

    DumpfeatureParams(dictionary,filename)

    return filename


def to_categorical(input_, num_classes):
    

    a = np.asarray([1 if i==input_-1 else 0 for i in range(num_classes)])
    return  a.reshape(1,num_classes)


def replace_with_unk(word,word2int):
    if word not in word2int:
        return 'UNK'
    else:
        return word

def convert_to_indices_and_pad(original_data_file,  words_and_indices_file, dataParams, mode = 'train'):
    

    
    data_file = mode + str('_features.h5')
    label_file = mode + str('_labels.h5')

    word2int = loadfeatureParams(words_and_indices_file)
    
    if mode == 'train':
        #extend vocabulary with "Pad token" and write new vocab dict into file
        word2int['PAD_TOKEN'] = dataParams['vocab_size']-1
        DumpfeatureParams(word2int, words_and_indices_file)

        print("added PAD_TOKEN to dict. current size of vocab: ", len(word2int))


    vocab_size = dataParams['vocab_size']

    max_review_length = dataParams['max_review_length']
    labels = []

    print("looping through reviews and converting to indices ... ")
    with h5py.File(data_file, "w") as ffile, h5py.File(label_file,"w") as lfile:

        #loop through every  review
        idx = 0
        neglect = 0
        with open(original_data_file) as f:
        
            for l in f:

                try:
                    l = eval(l)
                except:
                    continue

                curr_review_words = clean_str(l['Text'])

                curr_review_words_tokenized = list (map(  str.strip, curr_review_words.split() ))

                curr_review_words_tokenized = list (map( lambda x: replace_with_unk(x,word2int)  , curr_review_words_tokenized ))
                

                curr_review_words_indexed = list(map(lambda x: int(word2int[str(x)]), curr_review_words_tokenized ))

                curr_review_words_indexed.extend([vocab_size-1]*(max_review_length - len(curr_review_words_tokenized) ))

                # if(idx==0):
                #     print(curr_review_words_indexed)
                #     sys.exit()


                #print("len of curr_review_words_indexed: ", len(curr_review_words_indexed))
                if(len(curr_review_words_indexed) > max_review_length):
                    neglect += 1
                    continue

                assert len(curr_review_words_indexed)==max_review_length

                #convert to numpy array

                current_features = np.reshape(np.asarray(curr_review_words_indexed), (1,max_review_length))
                


                if idx==0:
                    dset = ffile.create_dataset("features", data = current_features ,
                                     shape = (1,max_review_length),
                                         maxshape=(None,max_review_length))

                    #create labels file

                    labels_array = to_categorical(l['Score'],5)

                    dset2 = lfile.create_dataset("labels", data = labels_array,
                                             shape = (1,5),
                                                 maxshape=(None,5))

                else:
                    dset.resize(dset.shape[0]+1,  axis=0)   
                    dset[-1:] = current_features

                    print("Features shape: ", dset.shape)

                    labels_array = to_categorical(l['Score'],5)

                    dset2.resize(dset2.shape[0]+1,  axis=0)
                    dset2[-1:] = labels_array

                    print("Labels shape: ", dset2.shape)






                idx+=1
                print("done with review %d" %idx)

       


    print("neglected %d samples due to high review length" %neglect)
               
    return data_file,label_file

    




    

def get_number_of_lines(file):

    with open(file) as f:
        return  sum(1 for _ in f)






def get_next_training_batch(batch_size, batch_index, featurefile, labelfile):
    
    i = batch_index
    modelParams = loadfeatureParams('modelParams.yaml')
    batchsize = modelParams['batch_size']

    with h5py.File(featurefile, "r") as ffile, h5py.File(labelfile,"r") as lfile:
        batch_x  = ffile['features'][i*batchsize:(i+1)*batchsize] 
        batch_y  = lfile['labels'][i*batchsize:(i+1)*batchsize] 

        
        # print("shape of batch_x: ",  batch_x.shape)

        # print("shape of batch_y: ",  batch_y.shape)

        return (batch_x, batch_y)


    





def get_validation_batch(num_valid_samples, batch_index, featurefile, labelfile):

    
    i = batch_index
    batchsize = num_valid_samples
    
    modelParams = loadfeatureParams('modelParams.yaml')
    batchsize = modelParams['batch_size']

    with h5py.File(featurefile, "r") as ffile, h5py.File(labelfile,"r") as lfile:
        batch_x  = ffile['features'][i*batchsize:(i+1)*batchsize] 
        batch_y  = lfile['labels'][i*batchsize:(i+1)*batchsize] 

        
        # print("shape of batch_x: ",  batch_x.shape)

        # print("shape of batch_y: ",  batch_y.shape)

        return (batch_x, batch_y)