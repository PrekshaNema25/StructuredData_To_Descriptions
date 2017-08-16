# Given the name of the model the corresponding to the argument passed 

# Usage
# $1 : The model that needs to be run
# $2 : The dataset for which the model needs to be run for:
# 	"french" "german" "weathergov" "english"


# Setting some hyperparamenters based on the dataset
# that has been passed

data=../data_
embedding=_embedding

mkdir -p ../output/

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
then
	tokens_per_field=5
	num_fields=15
	vocab_frequency_cutoff=15

else # german
	tokens_per_field=5
	num_fields=15
	vocab_frequency_cutoff=5

fi
 
if [ "$1" == "seq2seq" ]
then 
	python seq2seq_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300 --hidden-size 512 --batch-size 32 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --print_frequency 1000

elif [ "$1" == "hierarchy" ]
then
	python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300 --hidden-size 128 --batch-size 32 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb False --num-tokens-per-field $tokens_per_field --print_frequency 1000

elif [ "$1" == "nlb" ]
then
	python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300 --hidden-size 512 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000

elif [ "$1" == "mei_plus" ]
then 
	python mei_plus_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300 --hidden-size 128 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train False --gamma_tunable 10 --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000
fi


: '
Best hyperparameter settings for French Data:


if [ "$1" == "seq2seq" ]
then
        python seq2seq_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 256 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --print_frequency 1000

elif [ "$1" == "hierarchy" ]
        python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 512 --batch-size 32 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb False --num-tokens-per-field $tokens_per_field --print_frequency 1000

then
elif [ "$1" == "nlb" ]
then
        python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 256 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000

elif [ "$1" == "mei_plus" ]
then
        python mei_plus_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 256 --batch-size 32 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --gamma_tunable 5 --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000
fi



Best hyperparameter settings for German Data:


if [ "$1" == "seq2seq" ]
then
        python seq2seq_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 512 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --print_frequency 1000

elif [ "$1" == "hierarchy" ]
        python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 512 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb False --num-tokens-per-field $tokens_per_field --print_frequency 1000

then
elif [ "$1" == "nlb" ]
then
        python so_nlb_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 512 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000

elif [ "$1" == "mei_plus" ]
then
        python mei_plus_run_inference.py --work-dir $data$2/ --learning-rate 0.0004 --embedding-size 300
        --hidden-size 512 --batch-size 64 --epochs 0 --early_stop 5 --output/_dir ../output/ --emb-train
False --gamma_tunable 5 --vocab-freq $vocab_frequency_cutoff --num-fields $num_fields --feed-previous 20 --embedding-dir ../$2$embedding/ --is-stay-nlb True --num-tokens-per-field $tokens_per_field --print_frequency 1000
fi

'
