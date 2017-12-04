import tensorflow as tf 
import numpy as np 
from Params import loadfeatureParams,createParamsFile,DumpfeatureParams
import os,sys
import time
from CNN_model import TextCNN
from tensorflow.contrib import learn
from data_utilities import *
from tensorflow.contrib.tensorboard.plugins import projector

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



x_train_reviews_file = 'train_features.h5'
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

x_validation_reviews_file, y_validation_reviews_file =  convert_to_indices_and_pad(original_validation_file,
                                                                 words_and_indices_file,
                                                                dataParams, mode = 'validation' )


print("Done!")

total_training_samples = get_number_of_lines(original_train_file)

total_validation_samples = get_number_of_lines(original_validation_file)

total_test_samples = get_number_of_lines(original_test_file)


total_batches_per_epoch = int(70000/batch_size) -2



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

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#Training:

session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement)
sess = tf.InteractiveSession(config=session_conf)
tf.global_variables_initializer().run()


#sess = tf.Session(config=session_conf)


print("model")


sub_batch_size = modelParams['sub_batch_size']
num_sub_batches = modelParams['num_sub_batches']

valid_sub_batch_size = modelParams['valid_sub_batch_size']
valid_num_sub_batches = modelParams['valid_num_sub_batches']

print("monitoring training on %d training examples"%(sub_batch_size*num_sub_batches))
for epoch in range(epochs):


    for batch_index in range(total_batches_per_epoch):

        batch_x,batch_y = get_next_training_batch(batch_size, batch_index, x_train_reviews_file, 
                                                    y_train_reviews_file)

        #print(batch_x.shape)
        # print(batch_y)

        #print("running training step for batch %d"%batch_index)
        feed_dict = {cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob:dropout}
        sess.run(train_step, feed_dict= feed_dict)

        # acc, loss =   sess.run([cnn.accuracy, cnn.loss], 
        #                 feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob:dropout})

        # print("acc for batch: %f Loss for Batch %f"%(acc, loss))

        #print("done.")

    #print("[Training accuracy, loss] for epoch %s: " %epoch)

    print("done iterating through mini batches in this epoch")
    
    avg_acc = 0.0
    avg_loss = 0.0

    for i in range(num_sub_batches):
        train_subset_x,train_subset_y =  get_next_training_batch(sub_batch_size, i, x_train_reviews_file, 
                                                        y_train_reviews_file)

        acc, loss =   sess.run([cnn.accuracy, cnn.loss], 
                        feed_dict={cnn.input_x: train_subset_x, cnn.input_y: train_subset_y, cnn.dropout_keep_prob:dropout})
        
        avg_acc+=acc
        avg_loss+=loss

    avg_acc = 1.0*avg_acc/num_sub_batches
    avg_loss = 1.0*avg_loss/num_sub_batches
    
    print("Epoch: %d  Accuracy: %f  Loss:  %f " %(epoch, avg_acc, avg_loss)) 



    # code to visualize the embeddings. uncomment the below to visualize embeddings
    # run "'tensorboard --logdir='processed'" to see the embeddings
    final_embed_matrix = sess.run(cnn.W1)
    
    # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
    embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
    sess.run(embedding_var.initializer)

    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter('processed')
    summary_writer.add_graph(sess.graph)

    # # add embedding to the config file
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    
    # # link this tensor to its metadata file, in this case the first 500 words of vocab
    #embedding.metadata_path = 'processed/vocab_1000.tsv'

    # # saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, 'processed/model3.ckpt', 1)








    #print("predictions: ", preds)

    if(epoch%7 == 0):
       
        

        avg_acc = 0
        avg_loss = 0

        for i in range(valid_num_sub_batches):
            x_validation,y_validation = get_validation_batch(valid_sub_batch_size,i,
                                        x_validation_reviews_file, y_validation_reviews_file)



            #print("valid_sub_batch %d "%(i))

            acc, loss =   sess.run([cnn.accuracy, cnn.loss], 
                            feed_dict={cnn.input_x: x_validation, 
                                    cnn.input_y: y_validation, cnn.dropout_keep_prob:dropout})

            avg_acc+=acc
            avg_loss+=loss

        avg_acc = 1.0*avg_acc/valid_num_sub_batches
        avg_loss = 1.0*avg_loss/valid_num_sub_batches
            
        print("Epoch: %d  Validation Accuracy: %f  Validation Loss:  %f " %(epoch, avg_acc, avg_loss)) 
       








