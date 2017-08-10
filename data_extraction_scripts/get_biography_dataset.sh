mkdir -p ../biography_datatset

for i in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14"
#for i in "00" "01"
do
        wget -P ../biography_dataset/  https://github.com/DavidGrangier/wikipedia-biography-dataset/raw/master/wikipedia-biography-dataset.z$i
done


cat ../biography_dataset/wikipedia-biography-dataset.z?? > ../biography_dataset/tmp.zip

cd ../biography_dataset
unzip tmp.zip
rm tmp.zip

