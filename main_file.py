import tensorflow as tf 
import numpy as np 
from Params import loadfeatureParams,createParamsFile,DumpfeatureParams
import os,sys
import time
from CNN_model import TextCNN
from tensorflow.contrib import learn
from data_utilities import *


print("creating parameter file...")
createParamsFile()
modelParams = loadfeatureParams('modelParams.yaml')
dataParams = loadfeatureParams('dataParams.yaml')
globalParams = loadfeatureParams('globalParams.yaml')



print("loading variables...")
#load variables
dropout = modelParams['dropout']
learning_rate = modelParams['lr']
batch_size = modelParams['batch_size']
epochs = modelParams['epochs']
filter_sizes = modelParams['filter_sizes']
num_filters = modelParams['num_filters']
l2_reg = modelParams['l2_reg']
embedding_dim = modelParams['embedding_dim']
sequence_length = dataParams['max_review_length']
num_classes = modelParams['num_classes']
vocab_size = dataParams['vocab_size']
num_valid_samples = modelParams['num_valid_samples']



#original files:
# original_train_file = 'train_small.json' 
# original_test_file = 'test_small.json'
# original_validation_file = 'validation_small.json'


original_train_file = 'Train.json' 
original_test_file = 'Test.json'
original_validation_file = 'Validation.json'



words_and_indices_file = 'words_and_indices.yaml'


#build indexing dictionary 

if globalParams['build_vocab']:
    print("building indexing dict...")

    #dump that dict into a file
    words_and_indices_file = build_vocab_and_dump(dataParams, original_train_file)

    print("done!")



x_train_reviews_file = 'train_faetures.h5'
y_train_reviews_file = 'train_labels.h5'


x_validation_reviews_file = 'test_features.h5'
y_validation_reviews_file = 'test_labels.h5'



#convert word sentences into integers and pad. 
#(Returns the file path)


#create dataset and write to h5py:
if(globalParams['convert_and_pad']):

    print("converting to indices...")

    x_train_reviews_file, y_train_reviews_file =  convert_to_indices_and_pad(original_train_file, 
                                                                    words_and_indices_file,
                                                                dataParams )


    x_validation_reviews_file, y_validation_reviews_file =  convert_to_indices_and_pad(original_validation_file,
                                                                 words_and_indices_file,
                                                                dataParams, mode = 'validation' )


    # x_test_reviews_file, y_test_reviews_file =  convert_to_indices_and_pad(original_test_file,
    #                                                             words_and_indices_file,
    #                                                             dataParams, mode = 'test' )




print("Done!")

total_training_samples = get_number_of_lines(original_train_file)

total_validation_samples = get_number_of_lines(original_validation_file)

total_test_samples = get_number_of_lines(original_test_file)


total_batches_per_epoch = int(total_training_samples/batch_size) -2



modelParams['total_training_samples'] = total_training_samples

modelParams['total_validation_samples'] = total_validation_samples

modelParams['total_test_samples'] = total_test_samples


print("building Graph. Starting to train...")


cnn = TextCNN(
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocab_size= vocab_size,
            embedding_size= embedding_dim,
            filter_sizes= filter_sizes,
            num_filters= num_filters,
            l2_reg= l2_reg)



train_step = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss)


#Training:
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()








for epoch in range(epochs):


    for batch_index in range(total_batches_per_epoch):

        batch_x,batch_y = get_next_training_batch(batch_size, batch_index, x_train_reviews_file, 
                                                    y_train_reviews_file)

        feed_dict = {cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob:dropout}
        sess.run(train_step, feed_dict= feed_dict)

    print("[Training accuracy, loss] for epoch %s: " %epoch)
    acc, loss =   sess.run([cnn.accuracy, cnn.loss], 
                    feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob:dropout})
    print( acc, loss) 

    if(epoch%2 == 0):
       
        print("[Validation accuracy, loss] for epoch %s: " %epoch)

        x_validation,y_validation = get_validation_batch(num_valid_samples,
                                    x_validation_reviews_file, y_validation_reviews_file)

        acc, loss =   sess.run([cnn.accuracy, cnn.loss], 
                        feed_dict={cnn.input_x: x_validation, 
                                cnn.input_y: y_validation, cnn.dropout_keep_prob:dropout})
        print( acc, loss) 

       








