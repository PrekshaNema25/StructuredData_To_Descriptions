# Move On And Never Look Back: An Improved Neural Encoder Decoder Architecture for Generating Natural Language Descriptions from Structured Data

Given Wikipedia infoboxes for personalities generated natural language text


## Requirements:
* [tensorflow-0.12](https://www.tensorflow.org/versions/r0.12/get_started/os_setup)
* [gensim](https://pypi.python.org/pypi/gensim)
* [nltk](http://www.nltk.org/install.html)
* [matplotlib](https://matplotlib.org/users/installing.html)

## Data Download:
    The English Biography Dataset has been released as a part of the following work: .
    The dataset is avaliable at link : 
    You can also use the following command to download the dataset and convert it to the format
    required by the models given:
    sh data_extraction_scripts/get_biography_dataset.sh

## Pretrained Embeddings
    Download the pretrained embeddings for a given language and then store it to the destination folder. 
    * English Embedding: bash data_extraction_scripts/extract_embedding english english_embedding
    * French  Embedding: bash data_extraction_scripts/extract_embedding french french_embedding
    * German  Embedding: bash data_extraction_scripts/extract_embedding german german_embedding
    
    
## Various Proposed Model
    cd code
    * To run only the Seq2seq model:
      ./train.sh seq2seq
    * To run the hierarchy model:
       ./train.sh hierarchy 
    * To run the stay_on + never_look_back model:
        ./train.sh nlb
    * To run the mei++ model:
        ./train.sh mei_plus
   
    To tweak the hyperparameters for the above models please do the required changes in train.sh.
    The current configurations that have set are set based on the best performing hyperparameters
    during our experiments for English Wikibio Dataset. 
    

    
    
