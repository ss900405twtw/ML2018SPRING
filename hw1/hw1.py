import sys
import csv 
import math
import random
import numpy as np
import pandas as pd
 
filename=sys.argv[1]
df = pd.read_csv(filename,encoding="ISO-8859-1") 
test=df
# print(df.iloc[1:9,:])





w=np.array([0.001341796241341445, -0.017832882304246647, 0.02359433657921975, -0.024571467151734756, -0.04851484721118106, 0.07804869205974499, -0.07026158505965897, -0.09715176271191989, 0.21864155016638417, 0.014772512386577974, -0.0012239670603431143, 
    0.10101253863394145, -0.14548961124360069, 0.024160594230321807, 0.3538921112651885, -0.43789143900187205, 0.01733268410461687, 0.9408681867213817])
b=0.2594578794154284







ww=np.array([])
for k in range(0,4677,18):
    
    s=test.iloc[7+k,2:]
        
    # s=s.values

    # s=s.reshape(1,153)
    ww=np.append(ww,s)
        
ww=ww.astype('float')  

# print(ww[8])

data_x=ww.reshape(260,9)
for obj in data_x:
    for j in range(len(obj)):
        if j != 8 and (obj[j] <=0 or obj[j] >300) :
            obj[j]=obj[j+1]
        elif j==8 and (obj[j] <=0 or obj[j] >300):
            obj[j]=obj[j-1]
# print(data_x)



ww=np.array([])
for k in range(0,4677,18):
    s=test.iloc[8+k,2:]
    ww=np.append(ww,s)
        
ww=ww.astype('float') 
data_y=ww.reshape(260,9)
for obj in data_y:
    for j in range(len(obj)):
        if j != 8 and (obj[j] <=0 or obj[j] >300) :
            obj[j]=obj[j+1]
        elif j==8 and (obj[j] <=0 or obj[j] >300) :
            obj[j]=obj[j-1]

old=np.array([])
new=np.array([])
for i in range(260):
    old=np.append(data_x[i],data_y[i])
    # old=np.append(old,data_z[i])
    new=np.append(new,old)
new=new.reshape(260,9*2)
data_x=new


pm=[]
for j in range(260):
    sc=np.dot(w,(data_x[j].reshape(18,1)))+b
    sc=float(sc)
    pm.append(sc)
# print(len(pm))
output = sys.argv[2]



o=open(output,'a')
o.write("id")
o.write(",")
o.write("value")
o.write("\n")
for k in range(260):
    o.write("id_")
    o.write(str(k))
    o.write(",")
    o.write(str(pm[k]))
    o.write("\n")
o.close()
