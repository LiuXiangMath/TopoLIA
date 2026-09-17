import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import argparse

from configs import Para
from src import get_finetune_train_test_loader
from src import Finetune
from src import train_model,save_prediction,get_metrics

torch.set_default_dtype(torch.float32)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataname", type=int, default=2016)
    return parser.parse_args()



def prediction(para,year,seed,prediction_folder):
    # random seed
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled       = False
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    
    model_path = para.model_folder + '/' + para.model_name
    
    train_loader,test_loader = get_finetune_train_test_loader(year,para.finetune_batch_size,para.combination,para.num_statis,para.pdb_feature_folder,para.scaler_folder)
    model = Finetune(para,model_path=model_path).to(para.device)
    
    
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=para.finetune_lr,weight_decay=para.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=para.finetune_lr, steps_per_epoch=len(train_loader), epochs=para.finetune_epoch,pct_start=0.3)
    
    print(f'Start finetune on PDBbind-v{year} using seed {seed}')
    #print(para.print_attrs())
    

    for e in range(para.finetune_epoch):
        train_loss,train_pcc,train_rmse,model = train_model(model,train_loader,criterion,optimizer, scheduler,para.device)
        print(f'Epoch: {e+1}, Train Loss: {train_loss:.3f}, PCC: {train_pcc:.3f}, RMSE: {train_rmse:.3f}')
    save_prediction(model,test_loader,para.device,year,seed,e,para.finetune_epoch,prediction_folder)










if __name__ == "__main__":
    args = parse_args()
    para = Para()
    
    for seed in para.seeds:
        prediction(para,args.dataname,seed,para.save_folder)
    get_metrics(args.dataname,para.seeds,para.save_folder)




