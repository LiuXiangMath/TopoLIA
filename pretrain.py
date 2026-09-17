import sys
import time
import torch
import torch.nn as nn
import numpy as np
import random
import os

from configs import Para
from src import get_pretrain_loader
from src import Pretrain
from src import get_warmup_cosine_scheduler, save_model

torch.set_default_dtype(torch.float32)







def main(para):
    # random seed
    seed = para.pretrain_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled       = False
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    
    
    
    
    train_loader = get_pretrain_loader(para.pretrain_batch_size,para.combination,para.num_statis,para.pdb_feature_folder,para.scaler_folder)
    model        = Pretrain(para).to(para.device)
    
    
    base_lr         = para.pretrain_lr
    optimizer       = torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=para.weight_decay)
    steps_per_epoch = len(train_loader) 
    total_steps     = para.pretrain_epoch * steps_per_epoch
    warmup_steps    = int( para.warmup_rate * total_steps)
    scheduler       = get_warmup_cosine_scheduler(optimizer,warmup_steps=warmup_steps,total_steps=total_steps,min_lr_ratio=para.min_lr_ratio)
    
    print('Start:')
    #print(para.print_attrs())
    
    for e in range(para.pretrain_epoch):
        model.train()
        train_loss = 0.0
        for batch_index, data in enumerate(train_loader):
            data = data.to(para.device)
            
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss = train_loss + loss.item()
        
        print(f'Epoch: {e+1}, Loss: {train_loss:.3f}')
        if not os.path.isdir(para.model_folder):
            os.makedirs(para.model_folder)
        save_path = para.model_folder+f'/model-{e}.pt'
        remove_path = para.model_folder+f'/model-{int(e-1)}.pt'
        torch.save(model.state_dict(), save_path)
        os.system('rm '+remove_path)













if __name__ == "__main__":
    para = Para()
    main(para)
 

