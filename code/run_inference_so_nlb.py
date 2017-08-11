from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from optparse import OptionParser
from models.basic_files.dataset_iterator_hier import *
import os

class Config:

    """ Config class represents the hyperparameters in a single
        object
    """ 

    def __init__(self, learning_rate=0.0001, embedding_size=50, hidden_size=100,
               batch_size = 64,max_epochs = 20, max_sequence_length_content = 100, num_fields  = 20,
               max_sequence_length_title=50, early_stop=100, outdir="../out/",
               emb_tr=False, feed_previous = 5, vocab_frequency=74, embedding_dir = "../Data",is_stay_nlb = False, number_of_tokens_per_field = 5, print_frequency = 1000):

        """ Initialize the object with the parameters.

        Args:
            learning_rate : Learning rate for the optimizer
            embedding_size: dimensions of word embeddings
            hidden_size   : dimensions of hidden state of rnn cell
            batch_size    : batch size
            max_epochs    : Number of epochs to be run
            early_stop    : early stop

            max_sequence_length_content: Max length to be set for encoder inputs
            max_sequence_length_title  : Max length to be set for decoder inputs
        """


	print ("Config", emb_tr)
        config_file = open(outdir + "/config", "w")

        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.max_sequence_length_content = max_sequence_length_content
        self.max_sequence_length_field   = num_fields
        self.max_sequence_length_title   = max_sequence_length_title
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.outdir     = outdir
        self.is_stay_nlb = is_stay_nlb
	self.number_of_tokens_per_field = number_of_tokens_per_field
        self.emb_tr     = emb_tr
        self.early_stop = early_stop
	self.embedding_dir = embedding_dir
	self.vocab_frequency = vocab_frequency
	self.feed_previous = feed_previous
	self.print_frequency = print_frequency
        config_file.write("Learning rate " + str(self.learning_rate) + "\n")
        config_file.write("Embedding size " + str(self.embedding_size) + "\n")
        config_file.write("hidden size " + str(self.hidden_size) + "\n")
        config_file.write("Batch size " + str(self.batch_size) + "\n")
        config_file.write("Max Epochs" + str(self.max_epochs) + "\n")
        config_file.write("outdir " + str(self.outdir) + "\n")
        config_file.write("Early stop " + str(self.early_stop) + "\n")
        config_file.write("Embedding training" + str(self.emb_tr) + "\n")
        config_file.write("Feed Previous " + str(self.feed_previous) + "\n")
        config_file.close() 


class run_model:

    def __init__(self, wd, bA, config = None):

        """ The model is initializer with the hyperparameters.

            Args:
                config : Config() obeject for the hyperparameters.
        """

        # Use default hyperparameters
        if config is None:
            config = Config()

        self.config  = config
        self.model   = bA

        # Vocabulary and datasets are initialized.
        self.dataset = PadDataset(wd, self.config.embedding_size, self.config.vocab_frequency, embedding_dir = self.config.embedding_dir)


    def add_placeholders(self):

        """ Generate placeholder variables to represent input tensors
        """

        self.encode_input_placeholder  = tf.placeholder(tf.int32, shape=(self.config.max_sequence_length_content, None), name ='encode')
        self.decode_input_placeholder  = tf.placeholder(tf.int32, shape=(self.config.max_sequence_length_title, None),   name = 'decode')
        self.field_input_placeholder   = tf.placeholder(tf.int32, shape=(self.config.max_sequence_length_field, None),   name = 'field')
        self.sequence_length_input_placeholder   = tf.placeholder(tf.int64, shape=(self.config.max_sequence_length_field, None),   name = 'slength')
        self.label_placeholder         = tf.placeholder(tf.int32, shape=(self.config.max_sequence_length_title, None),   name = 'labels')
        self.weights_placeholder       = tf.placeholder(tf.int32, shape=(self.config.max_sequence_length_title, None),   name = 'weights')
        self.feed_previous_placeholder = tf.placeholder(tf.bool, name='feed_previous')

        #Could be used for dynamic padding
        #self.max_content_per_batch_p   = tf.placeholder(tf.int32, name='max_content')
        #self.max_title_per_batch_p     = tf.placeholder(tf.int32, name='max_title')


    def fill_feed_dict(self, encoder_inputs, decoder_inputs, labels, field, sequence_length,  weights, feed_previous=False):

        """ Fills the feed_dict for training at a given time_step.

            Args:
                encode_inputs : Encoder  sequences
                decoder_inputs : Decoder sequences
                labels : Labels for the decoder
                feed_previous : Whether to pass previous state output to decoder.

            Returns:
                feed_dict : the dictionary created.
        """

        feed_dict = {
        self.encode_input_placeholder : encoder_inputs,
        self.decode_input_placeholder : decoder_inputs,
        self.label_placeholder        : labels,
        self.field_input_placeholder  : field, 
        self.sequence_length_input_placeholder  : sequence_length, 
        self.weights_placeholder      : weights,
        self.feed_previous_placeholder: feed_previous,
        }

        return feed_dict


    def run_epoch(self, epoch_number, sess, fp = None):

        """ Defines the per epoch run of the model

            Args:
                epoch_number: The current epoch number
                sess:       :  The current tensorflow session.

            Returns
                total_loss : Value of loss per epoch

        """

        start_time = time.time()
        steps_per_epoch = int(math.ceil(float(self.dataset.datasets["train"].number_of_examples) / float(self.config.batch_size)))

        total_loss = 0

        for step in xrange(steps_per_epoch):

            # Get the next batch


            train_content, train_title, train_labels, train_field, sequence_length, train_weights, max_content, max_title= self.dataset.next_batch(
                self.dataset.datasets["train"],self.config.batch_size, True)



            """ Pass the decoder_inputs for the earlier epochs. As the model
                is trained, the outputs from the previous state should be fed
                to better train the model.
            """
            if (fp is None):
                if(epoch_number > self.config.feed_previous):
                    feed_previous = True
                else:
                    feed_previous = False

            else:
                feed_previous = fp

            feed_previous = False
            # Feed the placeholders with encoder_inputs,decoder_inputs,decoder_labels
            feed_dict = self.fill_feed_dict(train_content, train_title, train_labels, train_field, sequence_length, train_weights, feed_previous)


            #Minimize the loss
            outputs = sess.run(self.logits, feed_dict = feed_dict)
            _, loss_value = sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)
            total_loss  += loss_value

            duration = time.time() - start_time


	    #print ("Final outputs", len(outputs))
	    #print (sess.run(tf.shape(outputs[0])))
            #print('Trainable Variables') 
            #print ('\n'.join([v.name for v in tf.trainable_variables()]))

            #for v in tf.trainable_variables():
            #    x_shape = sess.run(v)
            #    print (x_shape.shape)


	    #x = sess.run(self.model.grad, feed_dict = feed_dict)	
            #print (x)

	    #print ("Loss value ", loss_value, " " , step)
            sys.stdout.flush()
            # Check the loss with forward propogation
            if (step + 1 == steps_per_epoch ) or ((step  + 1) % self.config.print_frequency == 0):

                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                sys.stdout.flush()
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.print_titles(sess, self.dataset.datasets["train"], 2)
                    
                # Evaluate against the validation set.
                print('Step %d: loss = %.2f' % (step, loss_value))
                print('Validation Data Eval:')
                #loss_value = self.do_eval(sess,self.dataset.datasets["valid"])
                self.print_titles(sess,self.dataset.datasets["valid"], 2)
                #print('Step %d: loss = %.2f' % (step, loss_value))
                sys.stdout.flush()

        return float(total_loss)/ float(steps_per_epoch)


    def do_eval(self,sess, data_set):

        """ Does a forward propogation on the data to know how the model's performance is.
             This will be mainly used for valid and test dataset.

            Args:
                sess : The current tensorflow session
                data_set : The datset on which this should be evaluated.

            Returns
                Loss value : loss value for the given dataset.
        """  

        total_loss = 0
        steps_per_epoch =  int(math.ceil(float(data_set.number_of_examples) / float(self.config.batch_size)))

        for step in xrange(steps_per_epoch): 
            train_content, train_title, train_labels, train_field, sequence_length, train_weights, max_content, max_title = self.dataset.next_batch(
                data_set,self.config.batch_size, False)
            
            feed_dict  = self.fill_feed_dict(train_content, train_title, train_labels,  train_field, sequence_length,  train_weights, feed_previous = True)
            loss_value = sess.run(self.loss_op, feed_dict=feed_dict)
            total_loss += loss_value

        return float(total_loss)/float(steps_per_epoch)



    def print_titles_in_files(self, sess, data_set, epoch):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        total_loss = 0
        awf  = []
        awt = []

        f1 = open(self.config.outdir + data_set.name + "_final_results" + str(epoch + 10 ), "wb")
        f2 = open(self.config.outdir + data_set.name + "_attention_weights" + str(epoch +10) , "wb")
        steps_per_epoch =  int(math.ceil(float(data_set.number_of_examples) / float(self.config.batch_size)))

        for step in xrange(steps_per_epoch):
            train_content, train_title, train_labels, train_field, sequence_length, train_weights, max_content, max_title = self.dataset.next_batch(
                data_set,self.config.batch_size, False)

            feed_dict = self.fill_feed_dict(train_content, train_title, train_labels, train_field, sequence_length, train_weights, feed_previous = True)

            _decoder_states_ , attention_weights, attention_weights_fields = sess.run([self.logits, self.attention_weights, self.attention_weights_fields], feed_dict=feed_dict)
            #_decoder_states_  = sess.run([self.logits], feed_dict=feed_dict)
            #print('attn_wt shape', np.shape(attention_weights))
            # print('dec_state shape', np.shape(_decoder_states_))
            #print ('attention_weights_fields', attention_weights_fields) 

            attention_states = np.array([np.argmax(i,1) for i in attention_weights])
           # Pack the list of size max_sequence_length to a tensor
            decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])
            awf.append(attention_weights_fields)
            awt.append(attention_weights)
            # tensor will be converted to [batch_size * sequence_length * symbols]
            ds = np.transpose(decoder_states)
            attn_state = np.transpose(attention_states)   
            true_labels = np.transpose(train_labels)
            # Converts this to a length of batch sizes
            final_ds = ds.tolist()
            final_as = attn_state.tolist()
            true_labels = true_labels.tolist()

            #print(final_ds)
            for i, states in enumerate(final_ds):

                # Get the index of the highest scoring symbol for each time step
                #indexes = sess.run(tf.argmax(states, 1))
                s =  self.dataset.decode_to_sentence(states)
                t =  self.dataset.decode_to_sentence(true_labels[i])
                f1.write(s + "\n")
                f1.write(t +"\n")
                x = " ".join(str(m) for m in final_as[i])
                f2.write(x + "\n")
        
        pickle.dump(awf, open(self.config.outdir + data_set.name + "_awf", "wb"))
        pickle.dump(awt, open(self.config.outdir + data_set.name + "_awt", "wb"))

    def print_titles(self, sess, data_set, total_examples):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """

        train_content, train_title, train_labels, train_field, sequence_length, train_weights, max_content, max_title = self.dataset.next_batch(
            data_set, total_examples, False)

        feed_dict = self.fill_feed_dict(train_content, train_title, train_labels,  train_field, sequence_length,  train_weights, feed_previous = True)

        _decoder_states_ = sess.run(self.logits, feed_dict=feed_dict)

        # Pack the list of size max_sequence_length to a tensor
        decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])

        # tensor will be converted to [batch_size * sequence_length * symbols]
        ds = np.transpose(decoder_states)
        true_labels = np.transpose(train_labels)

        # Converts this to a length of batch size
        final_ds = ds.tolist()
        true_labels = true_labels.tolist()

        for i,states in enumerate(final_ds):

            # Get the index of the highest scoring symbol for each time step
            #indexes = sess.run(tf.argmax(states, 1))
            print ("Title is " + self.dataset.decode_to_sentence(states))
            print ("True Summary is " + self.dataset.decode_to_sentence(true_labels[i]))



    def run_training(self):

        """ Train the graph for a number of epochs 
        """

        with tf.Graph().as_default():


            tf.set_random_seed(1357)

            self.config.max_sequence_length_content = max(val.max_length_content for i,val in self.dataset.datasets.iteritems())
            self.config.max_sequence_length_title = max(val.max_length_title for i,val in self.dataset.datasets.iteritems())


            len_vocab = self.dataset.length_vocab()
            initial_embeddings = self.dataset.vocab.embeddings

            self.add_placeholders()

            # Build a Graph that computes predictions from the inference model.
            self.logits, self.attention_weights, self.attention_weights_fields  = self.model.inference(self.encode_input_placeholder, self.decode_input_placeholder,
                                          self.field_input_placeholder, self.sequence_length_input_placeholder,  self.config.embedding_size,
                                          self.feed_previous_placeholder, len_vocab, self.config.hidden_size,
                                          weights = self.weights_placeholder, initial_embedding=initial_embeddings, 
                                          embedding_trainable=self.config.emb_tr, config=self.config)

            # Add to the Graph the Ops for loss calculation.
            self.loss_op = self.model.loss_op(self.logits, self.label_placeholder, self.weights_placeholder, len_vocab)

            # Add to the Graph the Ops that calculate and apply gradients.
            self.train_op = self.model.training(self.loss_op, self.config.learning_rate)


            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(self.config.outdir + "Logs" ,sess.graph)


            # if best_model exists pick the weights from there:
            if not (os.path.exists(self.config.outdir + "best_model.meta")):
                print ("Best model not saved, Inference cannot be done")
                return

            saver.restore(sess, self.config.outdir + 'best_model')
            test_loss = self.do_eval(sess, self.dataset.datasets["test"])

            print ("Test Loss:{}".format(test_loss))
            self.print_titles_in_files(sess, self.dataset.datasets["test"], 100)
            self.print_titles_in_files(sess, self.dataset.datasets["valid"], 100)

def main():
    parser = OptionParser()
 
    parser.add_option(
    "-w", "--work-dir", dest="wd", default="../Data/")
    parser.add_option(
        "-l", "--learning-rate", dest="lr", default=0.0001)
    parser.add_option(
        "-e", "--embedding-size", dest="emb_size",
        help="Size of word embeddings", default=50)
    parser.add_option(
        "-s", "--hidden-size", dest="hid_size",
        help="Hidden size of the cell unit", default=100)
    parser.add_option(
        "-a", "--batch-size", dest="batch_size",
        help="Number of examples in a batch", default=32)
    parser.add_option(
        "-n", "--epochs", dest="epochs",
        help="Maximum Number of Epochs", default=10)

    parser.add_option(
        "-t", "--early_stop", dest="early_stop",
        help="Stop after these many epochs if performance on validation is not improving", default=2)

    parser.add_option(
        "-o", "--output_dir", dest="outdir",
        help="Output directory where the model will be stored", default="../out/")
    (option, args) = parser.parse_args(sys.argv)
    c = Config(float(option.lr), int(option.emb_size), int(option.hid_size), int(option.batch_size), int(option.epochs), early_stop=int(option.early_stop), outdir= option.outdir)
    run_attention = run_model(option.wd, c)
    run_attention.run_training()

if __name__ == '__main__':
    main()
