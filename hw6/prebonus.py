import pandas as pd
import numpy as np
import csv
import keras
import sys



test=pd.read_csv('test.csv')
user_data=np.array(test['UserID'])
movie_data=np.array(test['MovieID'])


boys= keras.models.load_model('boys.hdf5')
girls=keras.models.load_model('girls.hdf5')
#model.summary()
#movie_emb = np.array(model.layers[1].get_weights()).squeeze()
#print("movie_emb shape: ",movie_emb.shape)
ans=np.zeros(len(user_data),dtype='float')
for i in range()
ans=model.predict([user_data,movie_data],batch_size=4096,verbose=1)



f = open(sys.argv[1],"w")
w = csv.writer(f)
w.writerow(('TestDataID','Rating'))

for i in range(len(ans)):    
    if(ans[i]>5):
        ans[i]=5.0
    if(ans[i]<1):
        ans[i]=1.0
    w.writerow((str(i+1),float(ans[i])))
f.close()
