import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import pickle
import os

torch.set_default_dtype(torch.float32)


# -------------------------------
# Dataset
# -------------------------------

def get_finetune_feature_label(year,combination,num_statis,folder,scaler_folder):
    bad = ['3tei',  '4as6']
    # label dict
    filename = './data/' + str(year) + '_INDEX_refined.data'
    f = open(filename)
    contents = f.readlines()
    f.close()
    pdb_label = {}
    start = {2007:5,2016:6,2013:6}[year]
    end = {2007:1305,2016:4063,2013:2965}[year]
    for i in range(start,end):
        tmp = contents[i]
        pdb = tmp[0:4]
        y = float(tmp[18:23])
        pdb_label[pdb] = y
    

    f = open('./data/train_data_' + str(year) + '.txt')
    train_pdb = eval(f.read())
    train_pdb = [p for p in train_pdb if p not in bad]
    f.close()
    
    f = open('./data/test_data_' + str(year) + '.txt')
    test_pdb = eval(f.read())
    test_pdb = [p for p in test_pdb if p not in bad]
    f.close()
    
    
    
    N = 100
    train_fea = np.zeros((len(train_pdb),N,combination,num_statis))
    train_label = np.zeros((len(train_pdb)))
    for i in range(len(train_pdb)):
        pdb = train_pdb[i]
        
        d = np.load(folder+'/'+pdb+'.npy')
        train_fea[i] = np.round(d[0:N],8)
        train_label[i] = pdb_label[pdb]
    
    test_fea = np.zeros((len(test_pdb),N,combination,num_statis))
    test_label = np.zeros((len(test_pdb)))
    for i in range(len(test_pdb)):
        pdb = test_pdb[i]
        
        d = np.load(folder+'/'+pdb+'.npy')
        test_fea[i] = np.round(d[0:N],8)
        test_label[i] = pdb_label[pdb]
    
    scaler_path = scaler_folder+'/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    train_fea = train_fea.reshape(-1,N*combination*num_statis)
    test_fea = test_fea.reshape(-1,N*combination*num_statis)
    train_fea = scaler.transform(train_fea)
    test_fea = scaler.transform(test_fea)
    train_fea = train_fea.reshape(-1,N,combination,num_statis).transpose(0, 2, 1, 3)
    test_fea = test_fea.reshape(-1,N,combination,num_statis).transpose(0, 2, 1, 3)
    
    return train_fea,train_label,test_fea,test_label
    


def get_pretrain_feature(combination,num_statis,folder,scaler_folder):
    f = open('./data/pretrain-pdbs.txt')
    train_pdb = eval(f.read())
    f.close()
    
    
    
    
    N = 100
    train_fea = np.zeros((len(train_pdb),N,combination,num_statis))
    for i in range(len(train_pdb)):
        pdb = train_pdb[i]
        d = np.load(folder+'/'+pdb+'.npy')
        train_fea[i] = np.round(d[0:N],8)
     
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    train_fea = train_fea.reshape(-1,N*combination*num_statis)
    train_fea = scaler.fit_transform(train_fea)
    train_fea = train_fea.reshape(-1,N,combination,num_statis).transpose(0, 2, 1, 3)
    
    if not os.path.isdir(scaler_folder):
        os.makedirs(scaler_folder)
    with open(scaler_folder+'/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_fea






class PretrainDataset(Dataset):
    """
    150*(20*8) persistent L0 of protein-ligand complex
    """
    def __init__(self,features):
        super().__init__()
        self.features = features
        #self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).to(torch.get_default_dtype())
        return x 
        

class FinetuneDataset(Dataset):
    """
    150*(20*8) persistent L0 of protein-ligand complex
    """
    def __init__(self,features,labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x,y = torch.from_numpy(self.features[idx]).to(torch.get_default_dtype()), torch.tensor([self.labels[idx]],dtype=torch.get_default_dtype())
        return x,y 


def get_pretrain_loader(batch_size,combination,num_statis,folder,scaler_folder):
    train_X = get_pretrain_feature(combination,num_statis,folder,scaler_folder)
    train_data = PretrainDataset(train_X)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=5, pin_memory=True)
    print('train:',len(train_data))
    return train_loader
    

def get_finetune_train_test_loader(year,batch_size,combination,num_statis,folder,scaler_folder):
    train_X,train_Y,test_X,test_Y = get_finetune_feature_label(year,combination,num_statis,folder,scaler_folder)
    train_data = FinetuneDataset(train_X,train_Y)
    test_data = FinetuneDataset(test_X,test_Y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,  num_workers=5, pin_memory=True)
    print('train:',len(train_data),'test:',len(test_data))
    return train_loader,test_loader
    

    
