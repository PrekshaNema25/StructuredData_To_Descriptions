import tensorflow as tf
import numpy as np
from mei_plus import *
from rnn_cell import *

import sys



class BasicAttention:

    """ Class Defines the basic attention model : 
        as defined in Paper : A neural attention model for abstractive text summarization
    """ 

    def add_cell(self,hidden_size, cell_input=None):

        """ Define the rnn_cell to be used in attention model

            Args:
                cell_input: Type of rnn_cell to be used. Default: LSTMCell
                hidden_size : Hidden size of cell
        """

        if(cell_input is None):
            self.enc_cell  = GRUCell(hidden_size)
        else:
            self.enc_cell = cell_input


    def add_projectionLayer(self, hidden_size, len_vocab):

        """ Add the projection layer for hidden_size x vocab

            Args:
                hidden_size : The hidden size of the cell
                len_vocab   : The number of symbols in vocabulary
        """
        self.projection_B = tf.get_variable(name="Projection_B", shape=[len_vocab])
        self.projection_W = tf.get_variable(name="Projected_W", shape=[hidden_size, len_vocab])


    def inference(self, encoder_inputs1, decoder_inputs1, query_inputs,field_inputs,sequence_length_inputs, \
    				embedding_size, feed_previous,
                  len_vocab, hidden_size, weights, embedding_trainable, config,  initial_embedding = None, c=None):

        """ Builds the graph for the basic attetion model

            Args:
                encoder_inputs: Placeholder for the encoder sequence
                decoder_inputs: Placeholder for the decoder sequence
                query_inputs  : Placeholder for the query   sequence
                embedding_size: Dimensions of the embedding for encoder and decoder symbols.
                feed_previous : Boolean to decide whether to feed previous state output 
                                to current rnn state for the decoder.
                len_vocab     : Number of symbols in encoder/decoder.
                hidden_size   : Hidden size of the cell state
                c             : The cell that needs to be used.
        
            Returns:
                A list of tensors of size [batch_size * num_symbols], that gives the
                probability distribution over symbols for each time step. The list
                is of size max_sequence_length
        """


        self.add_cell(hidden_size, c)
        self.add_projectionLayer(hidden_size, len_vocab)

        distract_cell = BasicLSTMCell(2*hidden_size, state_is_tuple = True)
        cell_encoder_bw = GRUCell(hidden_size)
        #enc_cell = DistractionLSTMCell(hidden_size)
        ei = tf.unpack(encoder_inputs1)
        di = tf.unpack(decoder_inputs1)
        qi = tf.unpack(query_inputs)
        fi = tf.unpack(field_inputs)
        si = tf.unpack(sequence_length_inputs)
        outputs, state, attention_weights_tokens, attention_weights_fields  = vad_seq2seq(encoder_inputs = ei,
                                                decoder_inputs = di,
                                                query_inputs = qi,
						field_inputs = fi, 
                                                cell_encoder_fw = self.enc_cell,
                                                cell_encoder_bw = cell_encoder_bw,
                                                distraction_cell = distract_cell,
					        sequence_field_length = si,
                                                embedding_trainable=embedding_trainable,
                                                num_encoder_symbols= len_vocab,
                                                num_decoder_symbols= len_vocab,
                                                embedding_size = embedding_size,
                                                output_projection= (self.projection_W, self.projection_B),
                                                feed_previous= feed_previous,
                                                initial_embedding = initial_embedding,
                                                dtype=tf.float32, 
					        num_fields=config.max_sequence_length_field)

        self.final_outputs = [tf.matmul(o, self.projection_W) + self.projection_B for o in outputs]

        print ("Attention weights fields size ", attention_weights_fields[0].get_shape())
        return self.final_outputs, attention_weights_tokens, attention_weights_fields


    def loss_op(self, outputs, labels, weights, len_vocab, attention_weights_fields, g):

        """ Calculate the loss from the predicted outputs and the labels

            Args:
                outputs : A list of tensors of size [batch_size * num_symbols]
                labels : A list of tensors of size [sequence_length * batch_size]

            Returns:
                loss: loss of type float
        """

        _labels = tf.unpack(labels)
        all_ones = [tf.ones(shape=tf.shape(_labels[0])) for _ in range(len(_labels))]
        weights = tf.to_float(weights)
        _weights = tf.unpack(weights)
        loss_per_batch = sequence_loss(outputs, _labels, _weights)

        loss_fields = tf.reduce_sum(attention_weights_fields[0], 1)
        loss_fields = tf.reduce_sum(tf.square(loss_fields - g))
        max_loss_fields = tf.reduce_sum(1 - tf.reduce_max(attention_weights_fields[0],1))

	print ("loss value", loss_fields.get_shape(), max_loss_fields.get_shape())

        if (g == 0):
           self.calculated_loss = loss_per_batch #+ loss_fields + max_loss_fields
        else:
           print ("the value of g is ", g)
           self.calculated_loss = loss_per_batch + loss_fields + max_loss_fields

        return self.calculated_loss


    def training(self, loss, learning_rate):

        """ Creates an optimizer and applies the gradients to all trainable variables.

            Args:
                loss : Loss value passed from function loss_op
                learning_rate : Learning rate for GD.

            Returns:
                train_op : Optimizer for training
        """

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        #self.grad = optimizer.compute_gradients(loss)
        return train_op


# To test the model
def main():
    n = basic_attention_model(c)
    n.inference(int(100))
    print "Inference"
    l = n.loss_op(n.final_outputs, int(100))
    print "Loss"
    n.training(l)
    print "Train"

if __name__ == '__main__':
    main()
