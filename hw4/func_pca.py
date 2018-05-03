import numpy as np
import csv 
import sys
from sklearn import decomposition
from sklearn.cluster import KMeans
import pandas as pd 
img = np.load(sys.argv[1])




img = img.astype('float32')/255
pca=decomposition.PCA(n_components=400,whiten=True,svd_solver="full").fit(img.T)
comp = pca.components_
print(comp.shape)

kmeans =KMeans(n_clusters = 2,random_state=0).fit(comp.T)
#cluster_labels = kmeans_fit.labels_




f=pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']),np.array(f['image1_index']),np.array(f['image2_index'])
  
o=open(sys.argv[3],'w')
o.write("ID,Ans\n")
for idx ,i1,i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else:
        pred =0
    o.write("{},{}\n".format(idx, pred))
o.close()
