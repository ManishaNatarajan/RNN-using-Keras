
# coding: utf-8

# # RUL Estimation using Health Index Curves - Altair_Submission - Manisha

# In[4]:


import numpy 
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
from keras.layers import Input, RepeatVector
from keras.models import Model
from math import sqrt


# In[5]:


# fix random seed for reproducibility
numpy.random.seed(7)


# In[6]:


#Dimensionality Reduction with PCA
def get_PCA(dataset):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=24)
    pca.fit(dataset)
    NCOMPONENTS = 1
    pca = PCA(n_components=NCOMPONENTS)
    pca_train = pca.fit_transform(dataset)
    #For plotting
#     print(dataset.shape)
#     print(pca_train.shape)
#     plt.plot(pca_train)
    return pca_train


# In[7]:


# convert an array of values into a dataset matrix -> used for Timeseries forecasting
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[8]:


#Obtain Health Index values for each train instance in Historical data
def get_HI_train():
    H = []
    for i in range(1,101):
        # load the dataset
        series = read_csv('Historical_Data.csv')
        series = series[series['unit number'] == i]
        series = series.drop(['unit number', 'RUL', 'time, in cycles'], axis = 1)
        dataset = series.values #Convert to np array
        dataset = dataset.astype('float32')#Convert to float
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        #Compute PCA
        dataset = get_PCA(dataset)
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # split into train and test sets
        train_size = int(len(dataset) * 0.2)
        test_size = len(dataset) 
        train, test = dataset[0:train_size,:], dataset[0:len(dataset),:]
#         print(len(train), len(test))
        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[:len(dataset)-2, :] = testPredict
        # plot baseline and predictions
#         plt.plot(scaler.inverse_transform(dataset))
#         plt.plot(trainPredictPlot)
#         plt.plot(testPredictPlot)
#         plt.show()

        #Compute Health Index
        diff = scaler.inverse_transform(dataset) - testPredictPlot
        diff = diff[:len(diff)-2]
        HI = scaler.fit_transform(diff)
        HI = numpy.vstack((HI,1))
        HI = numpy.vstack((HI,1))
        HI = 1 - HI
#         plt.plot(HI)
        H.append(HI)
        print(i)
    return H # return list of Health Index for train instances


# In[142]:


#Obtain Health Index values for each test instance
def get_HI_test(filename):
    H = []
    
    # load the dataset
    series = read_csv(filename)
#     series = series[series['unit number'] == i]
    series = series.drop(['unit number',  'time, in cycles'], axis = 1)
    dataset = series.values #Convert to np array
    dataset = dataset.astype('float32')#Convert to float
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #Compute PCA
    dataset = get_PCA(dataset)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.2)
    test_size = len(dataset) 
    train, test = dataset[0:train_size,:], dataset[0:len(dataset),:]
#         print(len(train), len(test))
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[:len(dataset)-2, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    #Compute Health Index
    diff = scaler.inverse_transform(dataset) - testPredictPlot
    diff = diff[:len(diff)-2]
    print(diff)
    plt.plot(diff)
    HI = (diff)
#     HI = numpy.vstack((HI,1))
#     HI = numpy.vstack((HI,1))
#     HI = 1 - HI
#         plt.plot(HI)
    H.append(HI)
        
    return H


# In[ ]:


H = get_HI_train()


# In[ ]:


H_test = get_HI_test('Engine1.csv')


# In[ ]:


Ht = numpy.absolute(H_test[0])
Ht = numpy.absolute(Ht - Ht[0])
Ht = 1 - Ht


# In[ ]:


def sim_comp(H, H_test):
    i = 0
    shifted_vals = []
    sim_vals = []
    while(i<len(H)):
        train_instance = H[i]
        
        if(len(train_instance)<len(H_test)):
            shifted_vals.append(0)
            sim_vals.append(10000)
            
        else:
            j = 0
            sim = 10000
            shift = 0
            while(j+len(H_test)<=len(train_instance)):
                diff = math.sqrt(mean_squared_error(train_instance[j:j+len(H_test)], H_test))
                if(diff<sim):
                    sim = diff
                    shift = j
                j=j+1
            shifted_vals.append(shift)
            sim_vals.append(sim)
            
        i = i+1
    return (shifted_vals, sim_vals)
            


# In[ ]:


s, ss = sim_comp(H, Ht)


# In[ ]:


index = ss.index(min(ss))


# In[ ]:


RUL = len(H[index])  - s[index] - len(Ht)


# In[ ]:


RUL


# In[ ]:


plt.plot(H[index], label='train instance')
#Shift the test instance for plotting
testPredictPlot = numpy.empty_like(H[index])
testPredictPlot[:, :] = numpy.nan
testPredictPlot[s[index]:len(Ht)+s[index], :] = Ht
plt.plot(testPredictPlot, label = 'test instance')
plt.xlabel('Cycles')
plt.ylabel('HI')
plt.title('Test instance 101')
plt.legend(loc = 'lower left')
plt.savefig('Test.png')
plt.show()

