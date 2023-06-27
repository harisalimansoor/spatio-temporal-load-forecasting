# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:26:46 2023

@author: Haris
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:29:11 2023

@author: Haris
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:53:25 2022

@author: Haris
"""

import numpy as np
import pandas as pd
import datetime

from dateutil import rrule
from datetime import date,datetime, timedelta
import networkx as nx

import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

#%%

#base_path='C:\\Users\\Haris\\Desktop\\spatio-temporal load forecasting paper\\LESCO Data from Dr Kashif\\'

base_path='F:/google drive/spatio-temporal load forecasting paper/LESCO Data from Dr Kashif/'

df = pd.read_csv(base_path+'LESCO Feeders weekly processed.csv')

print(df.isnull().sum().sum()) # check if null values

#df.to_csv(base_path+'LESCO Feeders hourly processed.csv',index=False)  

#%% plot data

"""
import matplotlib.pyplot as plt
import numpy as np

for i in range(1,len(df)):

    ypoints = np.array(df.iloc[:,i])
    
    fig=plt.figure()
    ax1 = fig.add_axes((0.1,0.4,0.8,0.5))
    ax1.set_title("This is my title"+str(i))
    ax1.plot(ypoints,label=str(i))
    plt.show()

"""
#%% removing anomalous data

#df=df.drop(['B.D-1', 'DHA PHASE - VII','BD-5','FEEDER-4 PHASE-V','FEEDER-9 PHASE-V'], axis=1)
#df=df.drop([365, 577])

[rows,columns]=df.shape

#%%

a=df.iloc[:,1:].corr()
a.values[[np.arange(len(a))]*2] = np.nan
a.columns=list(range(0,columns-1))
a=a.set_index([pd.Index(list(range(0,columns-1)))])

z=0.95

a[a > z] = 1

a[a < -z] = 1

a[(a > -z) & (a<z)] = 0
b1=a.stack().reset_index()
b1 = b1.loc[~((b1[0] == 0))] #remove zero rows        
    
b1.columns=['source','target','weight']
b1=b1.drop(columns=['weight'])
#b1 += 1    


#%%

df1 = df.copy(deep=True)

df1.columns=list(range(0,columns))

d=[]

x_df=[]
#n=[1,10,20,30,40,50,60]
#n=list(range(1,91))
n=list(range(1,2,1))
#n=[1,3]

m=len(n)

edges=b1
count1=0
count2=0
pasts=[4]
d=[]
f=[]
p=[]

for past in pasts:
    
    for count,k in enumerate(n):
    
        print(count)
        #for j in range(past+1,len(df1)-k):   
        for j in range(past+1,len(df1)-26):   
           # print(j)
            
            for i in range(1,columns):
                
                if ((i==1) and (count2!=0)):
                    
                    count1=count1+1
                    edges= pd.concat([edges, b1+(columns-1)*count1], axis=0,ignore_index=True)
                    
                count2=count2+1
                # print(i)
                dummy1 = np.zeros(m)
                dummy1[count] = 1
                
                #dummy2 = np.zeros(columns)
                #dummy1[count] = 1
                
                #c=[df1.iloc[j+k,0],dummy1,i,list(df1.iloc[j+k,41:48]),list(df1.iloc[j+k,48:60]),\
                 #  list(pd.concat([df1.iloc[j-past:j+1,i]])),df1.iloc[j+k,i]]
                
                    
                #x_df.append([df1.iloc[j+k,0]]+list(dummy1)+[i]+list(df1.iloc[j+k,41:48])+list(df1.iloc[j+k,48:60])+\
                   #list(df1.iloc[j-past:j+1,i])+list([df1.iloc[j+k,i]]))
                 
                x_df.append(list(dummy1)+list(df1.iloc[j-past:j+1,i])+list([df1.iloc[j+k,i]]))
                d.append(df.iloc[j+k,0]) #date of prediction
                f.append(i) #feeder number
                p.append(k) # future prediction number
                
                #c=[df1.iloc[j+k,0],dummy1,i,list(df1.iloc[j+k,41:48]),list(df1.iloc[j+k,48:60]),\
                #   list(pd.concat([df1.iloc[j-7:j+1,i]])),df1.iloc[j+k,i]]
                
                #d.append(c)
                
    df = pd.DataFrame (x_df, columns=list(range(0,past+m+2)))            

#%%


#df.to_csv(base_path+'weekly_pasts_4_future_12.csv')
#edges.to_csv(base_path+'weekly_edges_pasts_4_future_12_prob_0.95.csv')


#%%


#base_path='D:\\google drive\\spatio-temporal load forecasting paper\\LESCO Data from Dr Kashif\\'

#base_path='H:/My Drive/spatio-temporal load forecasting paper/LESCO Data from Dr Kashif/'
#n=list(range(1,26,1))
#edges = pd.read_csv(base_path+'weekly_edges_pasts_4_future_25_prob_0.95.csv')

#df = pd.read_csv(base_path+'weekly_pasts_4_future_25.csv')

def normalize_dataframe(Dataframe, Scaler):
    
    """
    scale a df if not scaled, Dataframe is the unscaled dataframe and scaler is 
    the class of sklearn, which you want to apply
    
    """

    x = Dataframe.values #returns a numpy 
    name=list(Dataframe.columns)
    
    x_scaled = Scaler.fit_transform(x)
    
    Dataframe = pd.DataFrame(x_scaled)
    Dataframe.columns = name
    return Dataframe

from sklearn import preprocessing

scaler_x = preprocessing.MinMaxScaler()

scaler_y = preprocessing.MinMaxScaler()

x_df=normalize_dataframe(df.iloc[:,0:-1], Scaler=scaler_x)

#y_df=normalize_dataframe(df.iloc[:,-1], Scaler=scaler_y)
#%%
y_df = scaler_y.fit_transform(np.array(df.iloc[:,-1]).reshape(-1,1))

#inv_transformed_y=scaler_y.inverse_transform(y_df)

y_df=pd.DataFrame(y_df) 
inv_transformed_y=scaler_y.inverse_transform(y_df)
#x_df=df_nor.iloc[:,0:-1]
#y_df=df_nor.iloc[:,-1]


#%%

from stellargraph import StellarGraph

def stellargraph_feeders(dataframe=None,dataframe_edge_list=None):
    
    #df = pd.read_csv(path_df)
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"])
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    #graph = StellarGraph({"players": df}, {"player_edges": df_edge_list})
    graph=StellarGraph(df, df_edge_list)
    
    print(graph.info())
    
    return graph

    
G=stellargraph_feeders(dataframe=x_df,dataframe_edge_list=edges)

#%% train test split

dummy=int(len(x_df)/len(n))

train_size=int(dummy*0.7)
vali_size=int(dummy*0.1)
test_size=int(dummy-train_size-vali_size)

train_x=[]
test_x=[]
vali_x=[]

train_y= pd.DataFrame([], columns=[0])
test_y=pd.DataFrame([], columns=[0])
vali_y=pd.DataFrame([], columns=[0])

train_date=[]
test_date=[]
vali_date=[]

train_feeders=[]
test_feeders=[]
vali_feeders=[]

train_p=[]
test_p=[]
vali_p=[]

for i in range(0,len(n)):
    
    train_x.extend(x_df.iloc[dummy*i:((dummy*i)+train_size),:].values.tolist())
    vali_x.extend(x_df.iloc[((dummy*i)+train_size):((dummy*i)+train_size+vali_size),:].values.tolist())
    test_x.extend(x_df.iloc[((dummy*i)+train_size+vali_size):dummy*(i+1),:].values.tolist())
    

    #train_y.extend(y_df.iloc[dummy*i:((dummy*i)+train_size)].values.tolist())
    #vali_y.extend(y_df.iloc[((dummy*i)+train_size):((dummy*i)+train_size+vali_size)].values.tolist())
    #test_y.extend(y_df.iloc[((dummy*i)+train_size+vali_size):dummy*(i+1)].values.tolist())
    train_y=pd.concat([train_y, y_df.iloc[dummy*i:((dummy*i)+train_size)]])
    vali_y=pd.concat([vali_y, y_df.iloc[((dummy*i)+train_size):((dummy*i)+train_size+vali_size)]])
    test_y=pd.concat([test_y, y_df.iloc[((dummy*i)+train_size+vali_size):dummy*(i+1)]])
    
    train_date.extend(d[dummy*i:((dummy*i)+train_size)][:])
    vali_date.extend(d[((dummy*i)+train_size):((dummy*i)+train_size+vali_size)][:])
    test_date.extend(d[((dummy*i)+train_size+vali_size):dummy*(i+1)][:])
    
    
    train_feeders.extend(f[dummy*i:((dummy*i)+train_size)][:])
    vali_feeders.extend(f[((dummy*i)+train_size):((dummy*i)+train_size+vali_size)][:])
    test_feeders.extend(f[((dummy*i)+train_size+vali_size):dummy*(i+1)][:])
    
    train_p.extend(p[dummy*i:((dummy*i)+train_size)][:])
    vali_p.extend(p[((dummy*i)+train_size):((dummy*i)+train_size+vali_size)][:])
    test_p.extend(p[((dummy*i)+train_size+vali_size):dummy*(i+1)][:])  
    

    
    
    
#%%

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from stellargraph.layer import GCN
from sklearn.metrics import mean_squared_error
# %matpltlib inline
"""
node_subjects=y_df

train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=int(len(y_df)*0.7), test_size=None, random_state=123
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=int(len(y_df)*0.1), test_size=None,random_state=123
)

train_subjects_x, test_subjects_x = model_selection.train_test_split(
    x_df, train_size=int(len(y_df)*0.7), test_size=None, random_state=123
)
val_subjects_x, test_subjects_x = model_selection.train_test_split(
    test_subjects_x, train_size=int(len(y_df)*0.1), test_size=None,random_state=123)
"""
#%%

generator = FullBatchNodeGenerator(G, method="gcn")

train_gen = generator.flow(train_y.index, train_y)
val_gen = generator.flow(vali_y.index, vali_y)
test_gen = generator.flow(test_y.index, test_y)    
    

#train_gen = generator.flow(train_subjects.index, train_subjects)

gcn = GCN(layer_sizes=[50], activations=["sigmoid"], generator=generator, dropout=0.5)

x_inp, x_out = gcn.in_out_tensors()

#x_inp2=layers.InputLayer(input_shape=(416,))
#x_inp2_layer1=layers.Dense(units=100, activation="sigmoid")(x_inp2)

#concatted = layers.Concatenate()([x_out, x_inp])

#pred1 = layers.LSTM(units=10, activation="relu")(x_out)

#f1=layers.Flatten(x_inp)

#concatted = layers.Concatenate()([x_inp2_layer1, pred1], axis=0)

pred2 = layers.Dense(units=50, activation="relu")(x_out)
pred22 = layers.Dense(units=10, activation="relu")(pred2)
#pred222 = layers.Dense(units=5, activation="relu")(pred22)
pred3 = layers.Dense(units=1, activation="sigmoid")(pred22)

#model = Model(inputs=[x_inp,x_inp2], outputs=pred3)
model = Model(inputs=[x_inp], outputs=pred3)

model.summary()

model.compile(
    optimizer='adam',
    loss=losses.mse,
    metrics=["mse"])

#val_gen = generator.flow(val_subjects.index, val_subjects)

#es_callback = EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)

es_callback = EarlyStopping( patience=20, restore_best_weights=True)


history = model.fit(
    train_gen,
    epochs=10000,
    validation_data=val_gen,
    verbose=2,
    batch_size=10,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    callbacks=[es_callback])

sg.utils.plot_history(history)

#test_gen = generator.flow(test_subjects.index, test_subjects)

predictions=model.predict(test_gen)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

#%%


embedding_model = Model(inputs=x_inp, outputs=x_out)
emb_train = np.squeeze(embedding_model.predict(train_gen))
emb_train.shape

emb_test = np.squeeze(embedding_model.predict(test_gen))
emb_test.shape

emb_val = np.squeeze(embedding_model.predict(val_gen))
emb_val.shape

x_emb_train=np.concatenate((np.array(train_x), emb_train), axis=1)
x_emb_test=np.concatenate((np.array(test_x), emb_test), axis=1)
x_emb_val=np.concatenate((np.array(vali_x), emb_val), axis=1)



#%% svR

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def svr(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
    regressor = SVR(kernel='rbf',C=1, epsilon=0.1)
    regressor.fit(train_xx,train_yy)
    #5 Predicting a new result
    yy_pred = regressor.predict(test_xx)  
  
    # Calculation of Mean Squared Error (MSE)
    #print('mse SVR:',mean_squared_error(test_yy,yy_pred))
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]

[_,a]=svr(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse SVR only GCN emb:',a)

[_,a]=svr(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse SVR GCN emb + x:',a)

[_,a]=svr(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse SVR GCN only x:',a)

#%% rf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
   
 # create regressor object
 
def rf(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):
    
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
     
    # fit the regressor with x and y data
    regressor.fit(train_xx,train_yy)
    #5 Predicting a new result
    yy_pred = regressor.predict(test_xx)  
  
    # Calculation of Mean Squared Error (MSE)
    #print('mse SVR:',mean_squared_error(test_yy,yy_pred))
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]

[_,a]=rf(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse RF only GCN emb:',a)

[_,a]=rf(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse RF GCN emb + x:',a)

[_,a]=rf(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse RF GCN only x:',a)


#%% gradient boosting


from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def gb(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):

    regressor = GradientBoostingRegressor(random_state=0)
    
    regressor.fit(train_xx,train_yy)
    #5 Predicting a new result
    yy_pred = regressor.predict(test_xx)  
  
    # Calculation of Mean Squared Error (MSE)
    #print('mse SVR:',mean_squared_error(test_yy,yy_pred))
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]

[_,a]=gb(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse GB only GCN emb:',a)

[_,a]=gb(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse GB GCN emb + x:',a)

[_,a]=gb(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse GB GCN only x:',a)



#%% NN

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset


def nn(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):
    
    
    model = Sequential()
    
    model.add(Dense(12, input_shape=(train_xx.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # fit the keras model on the dataset
    model.fit(train_xx, train_yy, epochs=30, batch_size=10)
    # evaluate the keras model
    yy_pred=model.predict(test_xx)
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]



[_,a]=nn(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse NN only GCN emb:',a)

[_,a]=nn(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse NN GCN emb + x:',a)

[_,a]=nn(train_xx=np.array(train_x),train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse NN only x:',a)
#%% XGB
import xgboost as xgb

def Xgb(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):

    regressor = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

    regressor.fit(train_xx,train_yy)
    #5 Predicting a new result
    yy_pred = regressor.predict(test_xx)  
  
    # Calculation of Mean Squared Error (MSE)
    #print('mse SVR:',mean_squared_error(test_yy,yy_pred))
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]

[_,a]=Xgb(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse XGB only GCN emb:',a)

[_,a]=Xgb(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse XGB GCN emb + x:',a)

[_,a]=Xgb(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse XGB only x:',a)



#%%
from keras.layers import LSTM

def lstm(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):
    
    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(10, input_shape=(train_xx.shape[1],1)))

    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # fit the keras model on the dataset
    model.fit(train_xx, train_yy, epochs=30, batch_size=10)
    # evaluate the keras model
    yy_pred=model.predict(test_xx)
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]


[_,a]=lstm(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse lstm only GCN emb:',a)

[_,a]=lstm(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse lstm GCN emb + x:',a)

[_,a]=lstm(train_xx=np.array(train_x),train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse lstm only x:',a)

#%% persistence forecast

from keras.layers import LSTM


def persistence(train_xx=None,train_yy=None,vali_xx=None, vali_yy=None, test_xx=None,test_yy=None):
    

    yy_pred=test_xx[:,5]
    
    return [yy_pred,mean_squared_error(test_yy,yy_pred)]


[_,a]=persistence(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)
 
print('mse persistence only GCN emb:',a)

[_,a]=persistence(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse persistence GCN emb + x:',a)

[_,a]=persistence(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse persistence only x:',a)


#%% arima

#%%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,mean_absolute_percentage_error

def error_measures(y_true=None, y_pred=None):
    
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred)
    exp_var=explained_variance_score(y_true,y_pred)
    mape= mean_absolute_percentage_error(y_true,y_pred)
    
    return [mae,mse,r2,exp_var, mape]
    
#%%    

import pandas as pd
  
# Creating Empty DataFrame and Storing it in variable df
results_df = pd.DataFrame()

results_df['date']=test_date
    
results_df['feeder']=test_feeders

results_df['ahead_prediction']=test_p

results_df['y_test']=list(test_y[0])

#%% SVR

[a1,a]=svr(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse SVR only GCN emb:',a)

results_df['y_SVR_GCN_only']=a1

[a1,a]=svr(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse SVR GCN emb + x:',a)

results_df['y_SVR_GCN_x']=a1

[a1,a]=svr(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse SVR GCN only x:',a)

results_df['y_SVR_x']=a1


#%% RF

[a1,a]=rf(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse RF only GCN emb:',a)
results_df['y_RF_GCN_only']=a1

[a1,a]=rf(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse RF GCN emb + x:',a)
results_df['y_RF_GCN_x']=a1

[a1,a]=rf(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse RF GCN only x:',a)
results_df['y_RF_x']=a1
#%% gb

[a1,a]=gb(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse GB only GCN emb:',a)
results_df['y_GB_GCN_only']=a1

[a1,a]=gb(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse GB GCN emb + x:',a)
results_df['y_GB_GCN_x']=a1

[a1,a]=gb(train_xx=train_x,train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse GB GCN only x:',a)
results_df['y_GB_x']=a1

#%% nn

[a1,a]=nn(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse NN only GCN emb:',a)
results_df['y_NN_GCN_only']=a1

[a1,a]=nn(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse NN GCN emb + x:',a)
results_df['y_NN_GCN_x']=a1

[a1,a]=nn(train_xx=np.array(train_x),train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse NN only x:',a)
results_df['y_NN_x']=a1
#%% lstm

[a1,a]=lstm(train_xx=emb_train,train_yy=train_y,test_xx=emb_test,test_yy=test_y)
 
print('mse lstm only GCN emb:',a)
results_df['y_LSTM_GCN_only']=a1

[a1,a]=lstm(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)

print('mse lstm GCN emb + x:',a)
results_df['y_LSTM_GCN_x']=a1

[a1,a]=lstm(train_xx=np.array(train_x),train_yy=train_y,test_xx=test_x,test_yy=test_y)

print('mse lstm only x:',a)
results_df['y_LSTM_x']=a1
#%% persistence

[a1,a]=persistence(train_xx=x_emb_train,train_yy=train_y,test_xx=x_emb_test,test_yy=test_y)
 
print('mse persistence only GCN emb:',a)
results_df['y_Persistence']=a1
#%% Error measures

errors_df = pd.DataFrame(columns=['mae','mse','r2','var','mape'])

#mae=[]
#mse=[]
#r2=[]
#var=[]
#mape=[]


for column in results_df.columns[4:]:
    #print(column)
    [mae1,mse1,r21,var1,mape1]=error_measures(y_true=scaler_y.inverse_transform(pd.DataFrame(results_df['y_test'])), y_pred=scaler_y.inverse_transform(pd.DataFrame(results_df[column])))

    errors_df.loc[column] = [float("{:.6f}".format(mae1)),float("{:.6f}".format(mse1)),float("{:.6f}".format(r21)),float("{:.6f}".format(var1)),float("{:.6f}".format(mape1))]

#mae.append(mae1)
#mse.append(mse1)
#r2.append(r21)
#var.append(var1)
#mape.append(mape1)

#%%

for column in results_df.columns[4:]:
    print(column)


#%% Data scaler

results_scaled=results_df.copy(deep=True)

for column in results_scaled.columns[3:]:
    #print(column)
    results_scaled[column] = scaler_y.inverse_transform(pd.DataFrame(results_df[column]))

#results_scaled.to_csv(base_path+"results.csv")

#%%

results2_df = pd.DataFrame()

for column in results_scaled.columns[0:4]:
    #print(column)
    results2_df[column] = results_scaled[column]

results2_df['y_SVR_GCN_x']=results_scaled['y_LSTM_x']
results2_df['y_RF_GCN_x']=results_scaled['y_SVR_x']
results2_df['y_GB_GCN_x']=results_scaled['y_RF_GCN_x']
results2_df['y_NN_GCN_x']=results_scaled['y_NN_x']
results2_df['y_LSTM_GCN_x']=results_scaled['y_GB_x']

results2_df['y_SVR_error']=results2_df['y_test']-results2_df['y_SVR_GCN_x']
results2_df['y_RF_error']=results2_df['y_test']-results2_df['y_RF_GCN_x']
results2_df['y_GB_error']=results2_df['y_test']-results2_df['y_GB_GCN_x']
results2_df['y_NN_error']=results2_df['y_test']-results2_df['y_NN_GCN_x']
results2_df['y_LSTM_error']=results2_df['y_test']-results2_df['y_LSTM_GCN_x']

#results2_df.to_csv(base_path+"results2.csv")
#%%

errors2_df = pd.DataFrame(columns=['mae','mse','r2','var','mape'])

for column in results2_df.columns[4:9]:
    #print(column)
    for j in range(1,40):
        
        [mae1,mse1,r21,var1,mape1]=error_measures(y_true=results2_df[results2_df['feeder'] ==j]['y_test'], y_pred=results2_df[results2_df['feeder'] ==j][column])

        errors2_df.loc[column+'_'+str(j)] = [float("{:.6f}".format(mae1)),float("{:.6f}".format(mse1)),float("{:.6f}".format(r21)),float("{:.6f}".format(var1)),float("{:.6f}".format(mape1))]


#errors2_df.to_csv(base_path+"errors2.csv")
#%%