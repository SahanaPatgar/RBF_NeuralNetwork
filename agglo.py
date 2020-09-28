
# coding: utf-8

# In[75]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import math
from sklearn.cluster import AgglomerativeClustering


# In[76]:


datapath_train=r"C:\Users\Meghana\Desktop\train"
datapath_test=r"C:\Users\Meghana\Desktop\test"
fruits=["apple","orange","banana"]


# In[79]:


train=[]
test=[]
for types in fruits:
            path=os.path.join(datapath_train,types)
            class_num=fruits.index(types)
            for img in os.listdir(path):
                        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                        img_array=cv2.resize(img_array,(100,100))
                        train.append([img_array,class_num])
                        plt.imshow(img_array,cmap="gray")
                        plt.show()
            path=os.path.join(datapath_test,types)
            class_num=fruits.index(types)
            for img in os.listdir(path):
                        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                        img_array=cv2.resize(img_array,(100,100))
                        test.append([img_array,class_num])
                        plt.imshow(img_array,cmap="gray")
                        #plt.show()





# In[80]:


print(len(train))
print(len(test))


# In[81]:


random.shuffle(train)


x_train=[]
y_train=[]

for i in train:
            x_train.append(np.array(i[0]).flatten())
            y_train.append(i[1])

            

            
x_test=[]
y_test=[]
for i in test:
            x_test.append(np.array(i[0]).flatten())
            y_test.append(i[1])                    
k=15


# In[82]:


hc=AgglomerativeClustering(n_clusters=15,affinity="euclidean",linkage="single")
hc.fit_predict(x_train)
print(hc.labels_)


# In[ ]:





# In[83]:


cluster=[]
for i in range(k):
    cluster.append([])


# In[84]:


j=0
for i in hc.labels_:
    cluster[i].append(x_train[j])
    j=j+1


# In[85]:


hcentroids=[]
for j in range(len(cluster)):
    hcentroids.append(np.mean(cluster[j], axis=0))
print(hcentroids)    


# In[86]:


for i in range(len(hcentroids)):
    plt.imshow(hcentroids[i].reshape(100,100),cmap="gray")
    plt.show()


# In[87]:


def convert_to_one_hot(y, num_of_classes):
        arr = np.zeros((len(y), num_of_classes))
        for i in range(len(y)):
            
            c =int(y[i] )
            arr[i][c] = 1
        return arr

    #calculate distance betweighteen data_set points and centroids
def get_distance(c,x): 
        sum = 0
        for i in range(len(c)):
            sum += (c[i] - x[i]) ** 2
        return np.sqrt(sum)


    #calculating gaussian function
    
def rbf(x, c, s):
        d=(float)(get_distance(c,x))
        return np.exp(-(float)(d**2) / ( s**2))


# In[ ]:





# In[88]:


max=0
for i in range(0,len(hcentroids)):
            for j in range(0,len(hcentroids)):
                d=get_distance(hcentroids[i],hcentroids[j])
            if(d>max):
                 max=d
d=max
std= d/math.sqrt(2*k) 
print(std)


# In[89]:


weight_h=[]
total_epochs=25
learning_rate=0.01
bias_h=[np.random.randn(3)]
weight_h=np.array([np.random.rand(3) for j in range(len(hcentroids))])
weight_h.reshape(3,k)   
print(weight_h.shape)
print(weight_h)
print(bias_h)


# In[90]:


y_train=convert_to_one_hot(y_train, 3)


# In[91]:


loss2=[]
epoch=0
for epoch in range(total_epochs):
    sum=0
    i=0
    for i in range(len(x_train)):
        a= np.array([rbf(x_train[i], c, std) for c in (hcentroids)]).reshape(1,k)
        F = a.dot(weight_h)+bias_h
        
        
        loss = np.sum((y_train[i] - F)** 2)
        sum+=loss/3
        error = np.array((y_train[i] - F)).reshape(1,3)
        weight_h = weight_h + learning_rate *(a.T).dot(error)
        bias_h = bias_h + learning_rate * error
    print("weight and bias after ", epoch+1)
    print(weight_h)
    print(bias_h)
    loss2.append(sum)
print('Loss occured:',loss2)


# In[92]:


c=0
y_pred = []
for i in range(len(x_test)):
        a = np.array([rbf(x_test[i], c, std) for c in hcentroids]).reshape(1,k)  
        F = a.dot(weight_h)+ bias_h
        y_pred.append(F)
        
print(y_pred)
       

y_pred= np.array([np.argmax(x) for x in y_pred])
print('prediction:',y_pred)
y_test=np.array(y_test)
print(y_test)
diff = y_pred - y_test
print(diff)
c=0
for i in range(len(diff)):
    if diff[i]==0:
        c=c+1
print('Accuracy from hierachical clustering: ', c/ len(diff))

