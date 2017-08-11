#1: Models will be saved here
#2 : The basic_dictionary
#3 : Queue
#4 : Number of epochs
#5: File run_script
#6: Number of fields
#7: Working directory 
#8: Vocab frequency 

#sh all_exp.sh domin_adaptaion nlb p8_7d 20 vad_hier_nlb_d 20 art_data 58
#sh all_exp.sh domain_adaptaion stay p8_7d 20 vad_hier_basic 20 art_data 58
#sh all_exp.sh domain_adaptation nlb p8_7d 20 vad_hier_nlb 20 sport_data 42
#sh all_exp.sh domain_adaptation stay p8_7d 20 vad_hier_basic 20 sport_data 42
#sh all_exp.sh domain_adaptation nlb p8_7d 20 vad_hier_nlb 20 politics_data 67


#sh all_exp.sh french_experiements french p8_7d 20 so_nlb_run_model.py 20 dat_french 15

mkdir ../$1
mkdir ../$1/hier_$2
mkdir ../$1/hier_$2/aaa_1
mkdir ../$1/hier_$2/aba_1
mkdir ../$1/hier_$2/aca_1
mkdir ../$1/hier_$2/baa_1
mkdir ../$1/hier_$2/bba_1
mkdir ../$1/hier_$2/bca_1

mkdir ../$1
mkdir ../$1/nlb_$2
mkdir ../$1/nlb_$2/aaa_1
mkdir ../$1/nlb_$2/aba_1
mkdir ../$1/nlb_$2/aca_1
mkdir ../$1/nlb_$2/baa_1
mkdir ../$1/nlb_$2/bba_1
mkdir ../$1/nlb_$2/bca_1


jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/aca_1/e1.txt -out ../$1/nlb_$2/aca_1/e1.txt python $5 -a 64 -s 512 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/aca_1/ -d ../french_embedding/ -k True -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/aaa_1/e1.txt -out ../$1/nlb_$2/aaa_1/e1.txt python $5 -a 64 -s 128 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/aaa_1/ -d ../french_embedding/ -k True -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/aba_1/e1.txt -out ../$1/nlb_$2/aba_1/e1.txt python $5 -a 64 -s 256 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/aba_1/ -d ../french_embedding/ -k True -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/baa_1/e1.txt -out ../$1/nlb_$2/baa_1/e1.txt python $5 -a 32 -s 128 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/baa_1/ -d ../french_embedding/ -k True -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/bba_1/e1.txt -out ../$1/nlb_$2/bba_1/e1.txt python $5 -a 32 -s 256 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/bba_1/ -d ../french_embedding/ -k True -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/nlb_$2/bca_1/e1.txt -out ../$1/nlb_$2/bca_1/e1.txt python $5 -a 32 -s 512 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/nlb_$2/bca_1/ -d ../french_embedding/ -k True -u 5 -c 1000



jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/aca_1/e1.txt -out ../$1/hier_$2/aca_1/e1.txt python $5 -a 64 -s 512 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/aca_1/ -d ../french_embedding/ -k False -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/aaa_1/e1.txt -out ../$1/hier_$2/aaa_1/e1.txt python $5 -a 64 -s 128 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/aaa_1/ -d ../french_embedding/ -k False -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/aba_1/e1.txt -out ../$1/hier_$2/aba_1/e1.txt python $5 -a 64 -s 256 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/aba_1/ -d ../french_embedding/ -k False -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/baa_1/e1.txt -out ../$1/hier_$2/baa_1/e1.txt python $5 -a 32 -s 128 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/baa_1/ -d ../french_embedding/ -k False -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/bba_1/e1.txt -out ../$1/hier_$2/bba_1/e1.txt python $5 -a 32 -s 256 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/bba_1/ -d ../french_embedding/ -k False -u 5 -c 1000
jbsubmit -queue $3 -proj french -cores 1+1 -mem 30g -require k80 -err ../$1/hier_$2/bca_1/e1.txt -out ../$1/hier_$2/bca_1/e1.txt python $5 -a 32 -s 512 -e 300 -l 0.0004 -n $4 -m $6  -t 5 -x 0 -p $8 -f 20 -w ../$7/ -o ../$1/hier_$2/bca_1/ -d ../french_embedding/ -k False -u 5 -c 1000
