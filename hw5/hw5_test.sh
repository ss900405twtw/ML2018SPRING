#!/bin/bash
wget https://www.dropbox.com/s/s498ejbl3xuhkzh/great0523084.hdf5
wget https://www.dropbox.com/s/jcnwl9pcge2ptvg/great0523086.hdf5
wget https://www.dropbox.com/s/hrqe62q18g53tmx/tok.pk
python3 pre0522.py $1 $2 $3
