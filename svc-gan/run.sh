logname=$1
cd bm
./train_v3.sh 2

cd ../em
./train_v3.sh 2
