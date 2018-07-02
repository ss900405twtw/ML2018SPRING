#!/bin/bash
cd src
wget https://www.dropbox.com/s/dw47e0rwhujzpay/model.zip
unzip model.zip
python3 hope0625.py
python3 0624_49.py
python3 49.py
python3 777.py
python3 reproduce_yuchi.py
python3 54387_reproduce.py
python3 54466_reproduce.py
python3 3ensemble_predict.py $1
mv predict.csv ../
rm *.csv 
cd ..

