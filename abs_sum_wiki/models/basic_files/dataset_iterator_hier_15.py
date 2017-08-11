import random
import nltk
import numpy as np
import pickle
import sys
import copy
import os.path
import tensorflow as tf
from vocab_hier_15 import *

class Datatype:

    def __init__(self, name,title,label, content, field, sequence_length_field, exm,max_length_content, max_length_title, max_field):

        """ Defines the dataset for each category valid/train/test

        Args:
            name : Name given to this partition. For e.g. train/valid/test
            title: The summarized sentence for the given source document
            content: Source documents
            exm:  Number of examples in this partition
            max_length_content : Maximum length of source documents among all examples
            max_length_title: Maximum length of title among all examples
            
            global_count_train: pointer to retrieve the next batch during training
            global_count_test : pointer to retrieve the next batch during testing
        """

        self.name = name
        self.title = title
        self.content = content
        self.labels = label
        self.field = field
        self.sequence_length_field = sequence_length_field
        self.number_of_examples = exm
        self.max_length_content = max_length_content
        self.max_length_title = max_length_title - 1
    
        self.max_field = max_field

        print (name, " " , max_length_content, " " , max_length_title)
        self.global_count_train = 0
        self.global_count_test = 0

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


    def make_batch(self, data, batch_size,count, max_length):

        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        batch = data[count:count+batch_size]
        count = count + batch_size
        #index = 0
        #temp = count + batch_size
        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            #index = index + 1
            count = 0
            
        batch = self.pad_data(batch,max_length)

        batch = np.transpose(batch)
        return batch, count

    def next_batch(self, dt, batch_size, c=True):

        #print("enter")
        if (c is True):
            count = dt.global_count_train
        
        else:
            count = dt.global_count_test


        max_length_content = max(val.max_length_content for i,val in self.datasets.iteritems())
        max_length_title   = max(val.max_length_title for i,val in self.datasets.iteritems())
        max_field          = max(val.max_field for i, val in self.datasets.iteritems())

        contents, count1 = self.make_batch(dt.content, batch_size,count, max_length_content)
        titles, _ = self.make_batch(dt.title, batch_size, count, max_length_title)
        labels, _ = self.make_batch(dt.labels, batch_size, count, max_length_title)

        field, _ = self.make_batch(dt.field, batch_size, count, max_field) 
        sequence_length_field, _ = self.make_batch(dt.sequence_length_field, batch_size, count, max_field) 
        weights = copy.deepcopy(titles)

        for i in range(titles.shape[0]):
            for j in range(titles.shape[1]):
                if (weights[i][j] > 0):
                        weights[i][j] = 1
                else:
                        weights[i][j] = 0

        if (c == True): 
            dt.global_count_train = count1 % dt.number_of_examples
        else:
            dt.global_count_test = count1 % dt.number_of_examples
        
        return contents, titles, labels, field, sequence_length_field,  weights, max_length_content, max_length_title
    
    def load_data_file(self,name, title_file, content_file, field_file, sequence_length_file):

        title = open(title_file,'rb')
        content = open(content_file,'rb')
        field = open(field_file, 'r')
        sequence_length = open(sequence_length_file, 'r')
        title_encoded = []
        content_encoded = []
        label_encoded = []
        field_encoded = []
        sequence_length_list = []

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


        max_field = 0
        for lines in field:
            temp = [self.vocab.encode_word(word) for word in lines.split()]
            if (len(temp) > max_field):
                max_field = len(temp)
            field_encoded.append(temp)
        print name


        max_l = 0
        for lines in sequence_length:
            temp = [int(word) for word in lines.split()]
            if (len(temp) > max_l):
                max_l = len(temp)

            sequence_length_list.append(temp)
           
        return Datatype(name, title_encoded, label_encoded, content_encoded, field_encoded, sequence_length_list, len(title_encoded), max_content, max_title, max_field)


    def load_data(self, wd="../Data/"):

        s = wd
        self.datasets = {}
        for i in ("train", "valid", "test"):
            temp_t = s + i + "_summary"
            temp_v = s + i + "_content"
            temp_f = s + i + "_field"
            temp_l = s + i + "_sequence_length"
            self.datasets[i] = self.load_data_file(i, temp_t, temp_v, temp_f, temp_l)


    def __init__(self,  working_dir = "../Data/", embedding_size=100, vocab_frequency = 74, global_count = 0, embedding_dir = "../Data"):
        filenames = [working_dir + "train_summary" , working_dir + "train_content", working_dir + "train_field", working_dir + "train_sequence_length"]
        #filenames = ["../DP_data/all_files"]
        self.global_count = 0
        self.vocab = Vocab()
        self.vocab.construct_vocab(filenames,embedding_size, vocab_frequency, embedding_dir)
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

def main():
    x = PadDataset([sys.argv[1],sys.argv[2]])
    x.load_data()
    print x.decode_to_sentence([2,1,2,3,4,5,8,7])
    for i in range(0,10):
        x.next_batch(x.datasets["train"], 2,True)
        x.next_batch(x.datasets["train"],2, False)

if __name__ == '__main__':
    main()
