import numpy as np 
import skimage
from skimage import io
import os 
import sys


	


img = np.zeros([415,600,600,3])
# print(sys.argv[1])
# for i in range(415):
# 	file = os.path.join(sys.argv[1],str(i))+'.jpg'
# 	# print (file)
# 	image[i]=io.imread(file)
# 	print(image[i].shape)
# i=0
# img = io.imread("jenni4.jpg")
# print(img)
for i in range(415):
	file= os.path.join(sys.argv[1],str(i))+'.jpg'
	img[i,:,:,:]=io.imread(file)

img=img.astype(np.uint8)
mean_img = np.mean(img ,axis =0)
print("mean: ",mean_img.shape)
#print("mean reshape: ",mean.reshape(-1,1).shape)
img_mean =img - mean_img

print(img_mean.shape)

#img_f = img.flatten()
#mean_f =mean.flatten()
img_mean= img_mean.reshape(415,-1)
img_mean = img_mean.T
print(img_mean.shape)
# print(x.shape)


#print average face
'''
mg = mean_img
mg -= np.min(mg)
mg /= np.max(mg)
mg =(mg*255).astype(np.uint8)
io.imsave('meanface.jpg',mg)
'''



U, s, V = np.linalg.svd(img_mean, full_matrices=False)
#find eigenvalue
print("s shape:",s.shape)
print("s sum: ",sum(s**2))
print("the eigen values are: ",s[:4])


#u = U[:,:4]
#print("u1.shape",u.shape)

#print eigen face

#p = u.reshape(600,600,3)
#p = p +mean_img
#p -= np.min(p)
#p /= np.max(p)
#p = (p*255).astype(np.uint8)

#io.imsave("eigenface_3.jpg",p)



#PRINT TOP 4 
print("U shape: ",U.shape)
#print(u.shape)
#u=U[:,:4].reshape([600,600,3,4])
u=U[:,:4]
#print("u2 shape",u.shape)
#u=u.reshape([-1,4])
#print(Uu.shape)
#print(u.shape)
#print(u = Uu)


file_address = os.path.join(sys.argv[1],sys.argv[2])
file_data = io.imread(file_address)
file_data = file_data - mean_img
file_data = file_data.reshape(-1,1)
weight = np.dot(u[:,:200].T,file_data)
M = np.dot(u[:,:200],weight)
M = M.reshape(600,600,3)+mean_img
M -= np.min(M)
M /= np.max(M)
M = (M*255).astype(np.uint8)
io.imsave('reconstruction.jpg',M)










