
# coding: utf-8

# In[1]:

import h5py
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger,ModelCheckpoint
import numpy as np
import sys,os
import pandas as pd
import datetime
from dateutil.parser import parse
from tqdm import tqdm
import pickle
# In[67]:

with open ('./data/scaleList', 'rb') as fp:
    scaleList = pickle.load(fp)
    
with open ('./data/scalerDic', 'rb') as fp:
    scalerDic = pickle.load(fp)   


# In[54]:

def getTrain(scaleList=scaleList,testMonth=8):
    originTrainXList = []
    originTrainYList = []
    originTestXList=[]
    originTestYList=[]
    
    for userData in tqdm(scaleList):
        #userData = pd.DataFrame(userData)
        trainX = userData[(userData.month < testMonth) & (userData.year == 2015)]
        trainY = userData[(userData.month < testMonth) & (userData.year == 2016)]
        testX = userData[(userData.month == testMonth) & (userData.year == 2015)]
        testY = userData[(userData.month == testMonth) & (userData.year == 2016)]
        
        train_XY = pd.merge(trainX,trainY,how='left',on=['month','day'])
        test_XY = pd.merge(testX,testY,how='left',on=['month','day'])
        
        originTrainXList.append(train_XY[['power_consumption_scale_x','week_x','max_x','min_x']])
        originTrainYList.append(train_XY[['power_consumption_scale_y','week_y','max_y','min_y']])
        originTestXList.append(test_XY[['power_consumption_scale_x','week_x','max_x','min_x']])
        originTestYList.append(test_XY[['power_consumption_scale_y','week_y','max_y','min_y']])
        
    with open('./data/originTrainXList', 'wb') as fp:
        pickle.dump(originTrainXList, fp)
    
    with open('./data/originTrainYList', 'wb') as fp:
        pickle.dump(originTrainYList, fp)
        
    with open('./data/originTestXList', 'wb') as fp: 
        pickle.dump(originTestXList, fp)
        
    with open('./data/originTestYList', 'wb') as fp:
        pickle.dump(originTestYList, fp)
 
        
    return originTrainXList,originTrainYList,originTestXList,originTestYList




# load
with open('./data/originTrainXList', 'rb') as fp:
    originTrainXList = pickle.load(fp)

with open('./data/originTrainYList', 'rb') as fp:
    originTrainYList = pickle.load(fp)

with open('./data/originTestXList', 'rb') as fp: 
    originTestXList = pickle.load(fp)

with open('./data/originTestYList', 'rb') as fp:
    originTestYList = pickle.load(fp)


# In[7]:

# LSTM  with window regression framing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import keras.backend.tensorflow_backend as KTF

def get_session(allow_growth=True):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    gpu_options = tf.GPUOptions(allow_growth=True)


    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
KTF.set_session(get_session(allow_growth=True))

# convert an array of values into a dataset matrix
def create_dataset(X, Y,look_back=7):
    dataX, dataY = [], []
    for i in range(len(X)-look_back):
        #print X.iloc[i:(i+look_back+1)]
        #print Y.iloc[i:(i+look_back+1)]
        # X consumption scale
        power_consumption_scale = X.iloc[i:(i+look_back+1)]['power_consumption_scale_x']
        featureX = power_consumption_scale.tolist() # consumption
        
        # weekBinary
        tmp = [0,0,0,0,0,0,0]
        weekNum = Y.iloc[i+look_back]['week_y']
        tmp[int(weekNum)] = 1
        featureX += tmp  #value
        
        # forecast day wether 
        wether_max = Y.iloc[(i+look_back)]['max_y']/30
        wether_min = Y.iloc[(i+look_back)]['min_y']/30
        featureX += [wether_max,wether_min]
        
        dataX.append(featureX)
        
        # Y consumption scale
        power_consumption_scale = Y.iloc[(i+look_back)]['power_consumption_scale_y']
        dataY.append([power_consumption_scale]) # consumption
        
        #print dataX
        #print dataY
        
    return np.array(dataX), np.array(dataY)
# dataX,dataY = create_dataset(originTrainXList[0],originTrainYList[0])


# In[5]:

# reshape into X=t and Y=t+1
import h5py
def trainData(ind=0,look_back=7):
    trainX, trainY = create_dataset(originTrainXList[ind],originTrainYList[ind],look_back=look_back)
    testX, testY = create_dataset(originTestXList[ind],originTestYList[ind],look_back=look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, trainX.shape[2]),))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # print  model.summary()
    
    
    # tensor board
    tb = TensorBoard('./data/tb/{}'.format(ind), histogram_freq=1,write_graph=True, write_images=False)
    #* tensorboard --logdir path_to_current_dir/Graph --port 8080 

    # earlystoping
    ES = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    # csv log
    csvlog = CSVLogger("./data/tflog/{}.log".format(ind),separator=',', append=False)

    # saves the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint(filepath="./data/weight/{}.h5".format(ind),verbose=0, save_best_only=True)

    ##############################
    # fit
    ##############################
    model.fit(trainX, trainY, epochs=50,batch_size=1, verbose=0,validation_data=(testX,testY),callbacks = [csvlog,checkpointer,ES])
    #model.fit(trainX, trainY, epochs=50,batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scalerDic[ind+1].inverse_transform(trainPredict.reshape(-1,1))
    trainY = scalerDic[ind+1].inverse_transform(trainY.reshape(-1,1))
    testPredict = scalerDic[ind+1].inverse_transform(testPredict.reshape(-1,1))
    testY = scalerDic[ind+1].inverse_transform(testY.reshape(-1,1))
    

for i in tqdm(range(len(originTrainXList))):
    trainData(ind=i)

