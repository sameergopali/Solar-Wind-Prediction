import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd




class SimpleDataset(Dataset):
    def __init__(self, X,y):
        self.X  = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return  len(self.X)

class SequenceDataset(Dataset):
    def __init__(self, X, target, sequence_length=80, num_features=11):
        data = X[np.newaxis,:,:]
        data = data.reshape(-1,sequence_length,num_features)
        self.y = torch.from_numpy(target.astype(np.float32))
        self.X =  torch.from_numpy(data.astype(np.float32))
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        x = self.X[index]
        y =  self.y[index]
        return x, y

def get_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler

def get_trainvaltest(X,y,val=True, test_size=0.3, val_size= 0.3,random_state=1):
    X_trainval, X_test , y_trainval , y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
    X_train, X_val , y_train , y_val = train_test_split(X_trainval,y_trainval,test_size=val_size, random_state=random_state)
    return X_train, X_val,  X_test , y_train, y_val, y_test

def get_sequence_dataloader(X_dataset, y_dataset,batchsize=64, test_size=0.3, val_size=0.3):
    data = X_dataset[np.newaxis,:,:]
    data = data.reshape(-1,80,11) 

    X_train, X_val,  X_test , y_train, y_val, y_test = get_trainvaltest(data, y_dataset, test_size=test_size, val_size=val_size)
    
    X_train = X_train.reshape(-1,11)
    X_val = X_val.reshape(-1,11)
    X_test = X_test.reshape(-1,11)


    scaler = get_scaler(X_train)
    Xtrain_scaled = scaler.transform(X_train)
    Xval_scaled   = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)


    train_set =  SequenceDataset(Xtrain_scaled, y_train)
    test_set = SequenceDataset(Xtest_scaled, y_test)
    val_set =  SequenceDataset(Xval_scaled, y_val)

    train_loader =  DataLoader(train_set, batch_size=batchsize,shuffle=True)
    test_loader =  DataLoader(test_set, batch_size=batchsize,shuffle=True)
    val_loader =  DataLoader(val_set, batch_size=batchsize,shuffle=True)
    
    return {"train_loader":train_loader, "test_loader":test_loader, "val_loader":val_loader}


def get_avg_dataloader(X_dataset, y_dataset,batchsize=64, test_size=0.3, val_size=0.3):

    X_train, X_val,  X_test , y_train, y_val, y_test = get_trainvaltest(X_dataset, y_dataset, test_size=test_size, val_size=val_size)
    scaler = get_scaler(X_train)
    Xtrain_scaled = scaler.transform(X_train)
    Xval_scaled   = scaler.transform(X_val)
    Xtest_scaled = scaler.transform(X_test)

    train_set =  SimpleDataset(Xtrain_scaled, y_train)
    test_set = SimpleDataset(Xtest_scaled, y_test)
    val_set =  SimpleDataset(Xval_scaled, y_val)

    train_loader =  DataLoader(train_set, batch_size=batchsize,shuffle=True)
    test_loader =  DataLoader(test_set, batch_size=batchsize,shuffle=True)
    val_loader =  DataLoader(val_set, batch_size=batchsize,shuffle=True)

    return {"train_loader":train_loader, "test_loader":test_loader, "val_loader":val_loader}

def get_dataset(pathX, pathY):
    df_x =  pd.read_pickle(pathX).to_numpy()
    df_y =  pd.read_pickle(pathY).to_numpy()
    return df_x, df_y

    