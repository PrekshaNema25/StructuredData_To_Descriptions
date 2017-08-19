import random
import nltk
import numpy as np
import pickle
import sys
import copy
import os.path
import tensorflow as tf
from vocab import *

class Datatype:

    def __init__(self, name,title,label, content, exm, max_length_content, max_length_title):

        """ Defines the dataset for each category valid/train/test

        Args:
            name : Name given to this partition. For e.g. train/valid/test
            title: The description of the input infobox.
	    label: Correct labels that will be used for loss computation 
            content: Input Infobox
            exm:  Number of samples in this dataset split
	    
            max_length_content : Maximum length of input Infoboxes among all samples
            max_length_title: Maximum length of description among all samples
            
            global_count_train: pointer to retrieve the next batch during training
            global_count_test : pointer to retrieve the next batch during testing
	    
	    Example for title, label:
	    	Description: <s> The apple is red <eos>
		title: <s> The apple is red
		label: The apple is red <eos>
        """

        self.name = name
        self.title = title
        self.content = content
        self.labels = label
        self.number_of_examples = exm
        self.max_length_content = max_length_content
	
	# As said in the given example the length of the title
	# will be one less than the length of the description
        self.max_length_title = max_length_title - 1
 

        self.global_count_train = 0
        self.global_count_test  = 0

class PadDataset:

    def find_max_length(self, data, count, batch_size):

        """ Finds the maximum sequence length for data of 
            size batch_size

            Args:
                data: The data from which sequences will be chosen
                count: The pointer from which retrieval should be done.
                batch_size: Number of examples to be taken to find max.

        """
        data = data[count:count + batch_size]
        return max(len(data[i]) for i,_ in enumerate(data))

    def pad_data(self,data, max_length):

        """ Pad the data to max_length given

            Args: 
                data: Data that needs to be padded
                max_length : The length to be achieved with padding

            Returns:
                padded_data : Each sequence is padded to make it of length
                              max_length.
        """

        padded_data = []

        for lines in data:
            if (len(lines) < max_length):
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=0)
            else:
                temp = lines[:max_length]
            padded_data.append(temp)

        return padded_data


    def make_batch(self, data, batch_size, idx, max_length):

        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                idx : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        batch = data[idx:idx+batch_size]
        idx = idx + batch_size
        #index = 0
        #temp = count + batch_size
        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            idx = 0
            
        batch = self.pad_data(batch,max_length)

        batch = np.transpose(batch)
        return batch, idx

    def next_batch(self, dt, batch_size, c=True):

        # c= True(False): Corresponds to the batch will be used for training(testing), 
	# Pick datapoints after the previously trained(tested) (batch)example.
	# idx denotes the index in the dataset
	if (c is True):
            idx = dt.global_count_train
        else:
            idx = dt.global_count_test

        max_length_content = self.datasets["train"].max_length_content
        max_length_title   = self.datasets["train"].max_length_title
 
	# idx_temp to account for the fact that the number of examples
	# might not be a multiple of batch_size. We need to keep track of overflow
	# from the number of datapoints.
        contents, idx_temp = self.make_batch(dt.content, batch_size, idx, max_length_content)
        titles, _ = self.make_batch(dt.title, batch_size, idx, max_length_title)
        labels, _ = self.make_batch(dt.labels, batch_size, idx, max_length_title)
 
	# titles contains the actual labels
	# we make a copy of titles to ensure that we calculate
	# loss for only those time steps which contain non pad symbols
        weights = copy.deepcopy(titles)

        for i in range(titles.shape[0]):
            for j in range(titles.shape[1]):
		# if weights[i][j] == 0 then the symbol is a pad
		# and we do not want to compute loss for that time step
		# else we compute the loss.
                if (weights[i][j] > 0):
                        weights[i][j] = 1
                else:
                        weights[i][j] = 0

        if (c == True): 
            dt.global_count_train = idx_temp % dt.number_of_examples
        else:
            dt.global_count_test = idx_temp % dt.number_of_examples
        
        return contents, titles, labels, weights, max_length_content, max_length_title

    def load_data_file(self,name, title_file, content_file):

        title = open(title_file,'rb')
        content = open(content_file,'rb')

        title_encoded = []
        content_encoded = []
        label_encoded = []

        max_title = 0
        for lines in title:

	    temp = [self.vocab.encode_word(word) for word in lines.split()]
            if (len(temp) > max_title):
                max_title = len(temp)
            title_encoded.append(temp[:-1])
            label_encoded.append(temp[1:])

        max_content = 0
        for lines in content:
            temp = [self.vocab.encode_word(word) for word in lines.split()]
            if (len(temp) > max_content):
                max_content = len(temp)
            content_encoded.append(temp)

        return Datatype(name = name, 
			title = title_encoded, 
			label = label_encoded,
			content = content_encoded,
			exm = len(title_encoded), 
			max_length_content = max_content, 
			max_length_title = max_title)

    def load_data(self, wd="../Data/"):

        s = wd
        self.datasets = {}
        for i in ("train", "valid", "test"):
            temp_t = s + i + "_summary"
            temp_v = s + i + "_content"
            self.datasets[i] = self.load_data_file(i, temp_t, temp_v)


    def __init__(self,  working_dir = "../Data/", embedding_size=100, vocab_frequency = 73,
		 embedding_dir = "../Data/", global_count = 0):
	
        filenames = [working_dir + "train_summary" , working_dir + "train_content"]
        #filenames = ["../DP_data/all_files"]

        self.global_count = 0
        self.vocab = Vocab()
        self.vocab.construct_vocab(filenames, embedding_size, vocab_frequency, embedding_dir)
        self.load_data(working_dir)

    def length_vocab(self):
        return self.vocab.len_vocab


    def decode_to_sentence(self, decoder_states):
        s = ""
        for temp in (decoder_states):
            if temp not in self.vocab.index_to_word:
                    word = "<unk>"
            else:
                word = self.vocab.decode_word(temp)
    
            s = s + " " + word
        return s

