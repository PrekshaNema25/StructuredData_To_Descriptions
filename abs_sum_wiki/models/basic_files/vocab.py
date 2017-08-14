import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer 
from gensim.models import Word2Vec
import gensim
from collections import defaultdict
from math import sqrt
import numpy as np 
import sys
class Vocab():
    def __init__(self):

        """ Initalize the class parameters to default values
        """

        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown       = "<unk>"
        self.end_of_sym    = "<eos>"
        self.start_sym     = "<s>"
        self.padding       = "<pad>"
        self.word_freq     = {}
        self.len_vocab     = 0
        self.total_words   = 0
        self.embeddings    = None


    def get_global_embeddings(self, filenames, embedding_size, embedding_dir):

        """ Construct the Embedding Matrix for the sentences in filenames.

            Args:
                filenames: File names of the training files.
                embedding_size: Dimensions for the embedding to be used.

            Returns
                Embedding matrix.
        """
        sentences = []

	if (os.path.exists(embedding_dir + 'vocab_len.pkl')):
		vocab_len_stored = pickle.load(open(embedding_dir + "vocab_len.pkl"))
	else:
		vocab_len_stored = 0

	if (vocab_len_stored == self.len_vocab and os.path.exists(embedding_dir + "embeddings.pkl")):
		print ("Load file")
		self.embeddings = pickle.load(open(embedding_dir +  "embeddings.pkl"))
		return None

        if (os.path.exists(embedding_dir + 'embeddings') == True):
            #model = gensim.models.KeyedVectors.load_word2vec_format('../Data/embeddings.bin', binary = True)
            #model = Word2Vec.load_word2vec_format('../Data/embeddings.bin', binary = True)
            model = gensim.models.KeyedVectors.load_word2vec_format(embedding_dir + 'embeddings', binary=False)
	    print ("Pretrained Embedding Loaded")
        else:
            for file in filenames:
                with open(file, 'rb') as f:
                    for lines in f:
                        words = [lines.split()]
                        sentences.extend(words)

            model = Word2Vec(sentences, size=embedding_size, min_count=0)
            model.save(embedding_dir + "embeddings")

        self.embeddings_model = model
        return model


    def add_constant_tokens(self):

        """ Adds the constant tokens in the dictionary.
        """

        self.word_to_index[self.padding]    = 0
        self.word_to_index[self.unknown]    = 1


    def add_word(self, word):

        """ Adds the word to the dictionary if not already present.

        Arguments:
            * word : Word to be added.

        Returns:
            * void
        """
        if word in self.word_to_index:
            self.word_freq[word] = self.word_freq[word] + 1

	elif word == "<pad>":
	    return
        else:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.word_freq[word] = 1
        
    def create_reverse_dictionary(self):

        """ Creates a mapping from index to the words
            This will be used during the time of decoding the
            indices to words.
        """

        for key, val in self.word_to_index.iteritems():
            self.index_to_word[val] = key

    def construct_dictionary_single_file(self, filename):
        
        """ Creates the dictionary from a single file.

            Arguments:
                * filename: The file from which dictionary
                          will be constructed

            Returns:
                * void
        """
        with open(filename, 'rb') as f:
            for lines in f:
                for words in lines.split():
                    self.add_word(words)


    def fix_the_frequency(self, limit=0):

        temp_word_to_index = {}
        temp_index_to_word = {}

        #get the list of the frequent words, upto the given limit.
        word_list = []
        count = 0

        new_index = 2
        for key in self.word_to_index:
            if (self.word_freq[key] > limit):
                temp_word_to_index[key] = new_index
                temp_index_to_word[new_index] = key
                new_index  = new_index + 1

        print new_index
        self.word_to_index = temp_word_to_index


    def construct_dictionary_multiple_files(self, filenames):

        """ Dictionary is made from all the files listed.

            Arguments :
                * filenames = List of the filenames 

            Returns :
                * None
        """

        for files in filenames:
            self.construct_dictionary_single_file(files)

    def encode_word(self, word):

        """ Conver the word to the particular index

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode_word(self, index):

        """ Index is converted to its corresponding word.

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_word:
            return self.unknown
        return self.index_to_word[index]


    def get_embeddings_pretrained(self, embedding_size, embedding_dir):

        """ Embeddings are appened based on the index of the 
        word in the matrix self.embeddings.
        """

        sorted_list = sorted(self.index_to_word.items(), key = operator.itemgetter(0))
        np.random.seed(1357)


	if (os.path.exists(embedding_dir + 'vocab_len.pkl')):
		vocab_len_stored = pickle.load(open(embedding_dir + "vocab_len.pkl"))
	else:
		vocab_len_stored = 0
	
	if vocab_len_stored == self.len_vocab and os.path.exists(embedding_dir + "embeddings.pkl"):
		self.embeddings = pickle.load(open(embedding_dir + "embeddings.pkl"))
		return

        embeddings = []
	count = 0
        for index, word in sorted_list:

            try:
                self.embeddings_model

                if word in self.embeddings_model.vocab:
			count = count + 1 
                        embeddings.append(self.embeddings_model[word])
                else:
                        if word in ['<pad>', '<s>', '<eos>']:
                                temp = np.zeros((embedding_size))

                        else:
                                temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size), (embedding_size))

                        embeddings.append(temp)

            except:
                if word in ['<pad>', '<s>', '<eos>']:
                    temp = np.zeros((embedding_size))
                else:
                    temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size), (embedding_size))

                embeddings.append(temp)

	print ("Number of words in the count" , count)
        self.embeddings = np.asarray(embeddings)
        self.embeddings = self.embeddings.astype(np.float32)


	pickle.dump(self.embeddings, open(embedding_dir + "embeddings.pkl", "w"))
	pickle.dump(self.len_vocab, open(embedding_dir + "vocab_len.pkl", "w"))


    def construct_vocab(self, filenames, embedding_size=100, vocab_frequency=74, embedding_dir="../Data/"):


        """ Constructs the embeddings and  vocabs from the parameters given.

            Args:
                * filenames: List of filenames to consider to make the vocab
                * embeddings: The size of the embeddings to be considered.

            Returns:
                * void
        """

        self.construct_dictionary_multiple_files(filenames)
        self.fix_the_frequency(vocab_frequency)
        #self.remove_the_unfrequent(10000)
        print "Length of the dictionary is " + str(len(self.word_to_index))
        sys.stdout.flush()
        self.add_constant_tokens()
        self.create_reverse_dictionary()
        self.get_global_embeddings(filenames, embedding_size, embedding_dir)
        self.get_embeddings_pretrained(embedding_size, embedding_dir)

        self.len_vocab = len(self.word_to_index)
        print "Length of the dictionary is " + str(len(self.word_to_index))
        self.total_words = float(sum(self.word_freq.values()))


    def plot_the_frequencies(self):
        
        x = self.index_to_word.keys()
        y = []

        for i in self.index_to_word.values():
            if i not in self.word_freq:
                y.append(0)
            else:
                y.append(self.word_freq[i])


        plt.hist(x, weights=y)
        plt.show()

def main():
    x = Vocab()
    filenames = ["../Gigaword/train_title", "../Gigaword/train_content"]
    x.construct_vocab(filenames)
    #x.plot_the_frequencies()

    "The vocab is " 
    #for i in x.word_to_index:
    #    print (i, x.word_freq[i])


if __name__ == '__main__':
    main()
