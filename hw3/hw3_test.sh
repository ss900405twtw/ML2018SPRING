#!/bin/bash
wget https://www.dropbox.com/s/wt6emo6n8qjdp3r/Model69.hdf5
wget https://www.dropbox.com/s/v8x8swaw5ptdrx3/Model.hdf5
python3 hw3_test.py $1 $2 $3
                                   
