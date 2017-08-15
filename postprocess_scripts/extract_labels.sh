# Clean the extracted labels by removing the <pad> and <eos> symbols.
# Post process the labels to also generate the copy predictions

#Usage:
#$1 : output_dir
#$2 : data_dir
#sh extract_labels $output_dir $data_dir

python retreive.py $1/test_final_results
python retreive.py $1/valid_final_results

python postprocess.py $2/test_content $1/test_final_results_plabels $1/test_attention_weights
python postprocess.py $2/valid_content $1/valid_final_results_plabels $1/valid_attention_weights

mkdir $1/predictions
mkdir -p $1/predictions/test
mkdir -p $1/predictions/valid

cp $1/test_final_results_plabels $1/predictions/test/
cp $1/test_final_results_plabels_copy $1/predictions/test


cp $1/valid_final_results_plabels $1/predictions/valid/
cp $1/valid_final_results_plabels_copy $1/predictions/valid/
