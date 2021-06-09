#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset


# In[4]:


cons=ts.get_apis()
df=ts.bar(
    code='000300',
    conn=cons,
    start_date='2015-01-01',
    end_date='',
    asset='INDEX',
    retry_count=3,
)


# In[40]:


def generate_data_by_n_days(series,n):
    
    '''
        function:generate a dataframe by rolling window
        args:
            series(series/list):the list we need to roll
            n(int):the size of rolling window
        return:
            df(dataframe) 
    '''
    
    if len(series)<=n:
        raise Exception("the length is not enough")
    df=pd.DataFrame()
    for i in range(n):
        df["c%d"%i]=series.tolist()[i:-(n-i)]
    df['y']=series.tolist()[n:]
    return df


# In[41]:


def genertate_train_test_set(rowdata,trainratio=0.7):
     
    '''
        function:generate train_test set
        args:
            rowdata(dataframe): the rawdata
            trainratio(float):len(train_set)/len(test_set)
        return:
            train_set(dataframe)
            test_set(dataframe)
    '''
    
    train_len=int(trainratio*len(rowdata))
    train_set=rowdata[:train_len]
    test_set=rowdata[train_len+1:]
    return train_set,test_set


# In[42]:


def data_nomalization(df):
    
    '''
        function:data nomalization and transform the data to tensor
        args:
            df(dataframe):the data we need to clean
        return:
            train_set(dataframe)
            test_set(dataframe)
    '''
    
    df.dropna()
    df_numpy=np.array(df)
    df_numpy=(df_numpy-np.mean(df_numpy))/np.std(df_numpy)
    return torch.Tensor(df_numpy)


# In[43]:


class RNN(nn.Module):
    
    '''
        class:one-layer lstm and a full-connected network
        input:
            tensor(30*1)
        output:
            tensor(1*1)
    '''
    
    def __init__(self,input_size):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out=nn.Sequential(
            nn.Linear(64,1)
        )
    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.out(r_out)
        return out


# In[45]:


feature_size=30
rnn=RNN(feature_size)
optimizer=torch.optim.Adam(rnn.parameters(),lr=0.1)
raw_data=generate_data_by_n_days(df['close'],30)
train_set,test_set=genertate_train_test_set(raw_data)
train_set=data_nomalization(train_set)
test_set=data_nomalization(test_set)
print(train_set.shape)
print(test_set.shape)
trainloader=DataLoader(train_set,batch_size=20,shuffle=False)
loss_function=nn.MSELoss()


# In[87]:


def train(lose_function):
    
    '''
        function:train the dataset with different loss function
        args:
            loss_function:the loss_function we selected
        return:
            loss_list:the loss when we train
    '''
    
    loss_list=[]
    for step in range(500):
        for tx in trainloader:
            #print(tx.shape)
            features=torch.unsqueeze(tx,dim=0)
            output=rnn(features[:,:,0:30])
            loss=loss_function(torch.squeeze(output),features[:,:,30])
            loss_list.append(float(loss.data))
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
    return loss_list


# In[89]:


plt.plot(train(nn.MSELoss))


# In[92]:


plt.plot(train(nn.MAELoss))


# In[94]:


plt.plot(train(SmoothL1Loss))


# In[ ]:




