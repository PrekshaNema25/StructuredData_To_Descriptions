# NLG_From_StructuredData
Given Wikipedia infoboxes for personalities generated natural language text


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
   
    To tweak the hyperparameters for the above models please do the required changes in train.sh. The current configurations
    that have set are set based on the best performing hyperparameters during our experiments for English Wikibio Dataset. 
    
    
