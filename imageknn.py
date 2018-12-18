#%%
import os, codecs
import shutil
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from keras.applications.vgg16 import decode_predictions 
from keras.preprocessing.image import img_to_array
import csv
import pandas as pd
import keras
import matplotlib.pyplot as plt
import xlrd
vgg16_model = VGG16(include_top=False, weights="imagenet", input_shape=(64,64,3))

resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(64,64,3))
def get_file_name(path):
    '''''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    #print(filenames)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)
    return path_filenames
def preprocess(filenames):
    model = VGG16()
    
    im=[]
    i=0
    #print(model.summary())
    for file in filenames:
        image = cv2.imread(file)
        # Resize it to 224 x 224
        image = cv2.resize(image, (64,64))
        #print(image.shape,type(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im.append(image)
        #print(im)
        images = np.array(im, dtype=np.float32)
        images /= 255
    #print(images.shape)
    #features = vgg16_model.predict(images)
    features=resnet50_model.predict(images)
    features = features.reshape(images.shape[0], -1)
    return features

def knn_detect(features, randomState=None):

    SSE=[]
    K=[]
    for k in range(10, 28): 
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        SSE.append(kmeans.inertia_) 
        K.append(k)
    X = range(10, 28) 
    plt.xlabel('k') 
    plt.ylabel('SSE') 
    plt.plot(X, SSE, 'o-') 
    plt.show()
    sse=min(SSE)
    i=SSE.index(sse)
    print(K[i])
    #print(value)

    return K[i]
def res_fit(filenames, labels): 
    files = [file.split('/')[-1].replace('.jpg','') for file in filenames] 
    
    for i in range(len(labels)):
        shutil.copy('/home/joker/5002/image/'+files[i]+'.jpg', '/home/joker/5002/re2/'+str(labels[i])+files[i])
    return dict(zip(files, labels))

def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))
def main(): 
    path_filenames = get_file_name("/home/joker/5002/image/") 
    features=preprocess(path_filenames)

    k= knn_detect(features)
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    labels, cluster_centers=kmeans.labels_, kmeans.cluster_centers_
    #for file in path_filenames:
    #    file=file.replace('.jpg','')
    res_dict = res_fit(path_filenames, labels)
    q=sorted(res_dict.items(),key=lambda item:item[1])
    #print(q)
    ll=[]
    for i in range(k):
        el=[]
        for e in range(len(q)):
            if q[e][1]==i:
                el.append(q[e][0])
        ll.append(el)
    
    ll=pd.DataFrame(ll)
    lc=pd.DataFrame()
    for indexs in ll.index:
        lc[indexs]=ll.loc[indexs].values[0:-1]
    for col in lc.columns:
        for i in range(len(lc[col])):
            
            lc[col][i]="'"+str(lc[col][i])+"'"
    lc=lc.replace("'None'",'')
    lc.columns=['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8','Cluster 9','Cluster 10','Cluster 11','Cluster 12','Cluster 13','Cluster 14','Cluster 15','Cluster 16','Cluster 17','Cluster 18','Cluster 19','Cluster 20','Cluster 21','Cluster 22','Cluster 23','Cluster 24','Cluster 25','Cluster 26','Cluster 27']
    #print(lc.shape)
    lc.to_csv('A3_xmaoad_20548149_prediction.csv',index=None)
        
if __name__ == "__main__": 
    main()



