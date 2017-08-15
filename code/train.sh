# Given the name of the model the corresponding to the argument passed 

# Usage
# $1 : The model that needs to be run
# $2 : The dataset for which the model needs to be run for:
# 	"french" "german" "weathergov" "english"


# Setting some hyperparamenters based on the dataset
# that has been passed

if [ "$2" == "weathergov" ]
then
	tokens_per_field=10
	num_fields=20
	vocab_frequency_cutoff=0

elif [ "$2" == "english" ]
then
	tokens_per_field=5
	num_fields=20
	vocab_frequency_cutoff=74

elif [ "$2" == "french" ]
	tokens_per_field=5
	num_fields=15
	vocab_frequency_cutoff=15

else # german
	tokens_per_field=5
	num_fields=15
	vocab_frequency_cutoff=5

fi
 
tokens_per_field = 5
if [ "$1" == "seq2seq" ]
then 
	python run_training_vad.py

elif [ "$1" == "hierarchy" ]
then 
elif [ "$1" == "nlb" ]
then
elif [ "$1" == "mei_plus" ]
then 
fi

