import numpy as np 
import pandas as pd 
import sys


s=pd.read_csv('54466.csv')

#s1=pd.read_csv('777.csv')
#s2=pd.read_csv('777.csv')
#s1=pd.read_csv('window15.csv')
#s2=pd.read_csv('.csv')
#s1=pd.read_csv('49.csv')
s1=pd.read_csv('54387.csv')
#s1=pd.read_csv('0617_dualencoder_test_yuchi_0.47905.csv')
#s4=pd.read_csv('.csv')
s2=pd.read_csv('0614_dualencoder_test1_yuchi.csv')
#s2=pd.read_csv('777.csv')
#s4=pd.read_csv('06.csv')


#s2=pd.read_csv('0615_embedding_test3_yuchi_0.44937.csv')


ss=list(s['ans'])
s11=list(s1['ans'])
s22=list(s2['ans'])
#s33=list(s3['ans'])
#s44=list(s4['ans'])
# input(ss)
ans=np.zeros(5060)

for i in range(5060):
	array=[]
	array.append(ss[i])
	array.append(s11[i])
	array.append(s22[i])
#	array.append(s33[i])
#	array.append(s44[i])
	# array=ss[i]+s11[i]+s22[i]
	# print(array)
	# num=np.argmax(counts)
	if len(np.unique(array))==3:
		ans[i]=ss[i]
#	elif len(np.unique(array))==4:
#		ans[i]=ss[i]
	else:
		# print(i)
		counts = np.bincount(array)
		ans[i]=np.argmax(counts)
print(ans)
ans=ans.astype('int')
o=open('predict.csv','a')
o.write("id")
o.write(",")
o.write("ans")
o.write("\n")
for k in range(5060):
    #o.write("id_")
    o.write(str(k))
    o.write(",")
    o.write(str(ans[k]))
    o.write("\n")
o.close()
