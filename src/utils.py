import math
from torch.optim.lr_scheduler import LambdaLR
import os
import random
import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error
import torch


# -------------------------------------------------------------------------------------------------
# warmup scheduler, save and load model
# -------------------------------------------------------------------------------------------------
def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        current_step += 1 
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)  
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def save_model(path, model, para, optimizer, scheduler, epoch):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'rng_state': get_rng_state(),
        'para':para,
        
    }
    torch.save(ckpt, path)

def load_model(path, model, optimizer=None, scheduler=None,map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])

    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])


    if 'rng_state' in ckpt and ckpt['rng_state'] is not None:
        set_rng_state(ckpt['rng_state'])

    return ckpt['epoch']
    

# -------------------------------------------------------------------------------------------------
# Train / evaluate
# -------------------------------------------------------------------------------------------------
def train_model(model, dl, criterion, optimizer, scheduler,device):
    model.train()
    total_loss, total = 0, 0
    true_y,pred_y = [],[]
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        
        logits_ = logits.view(-1)
        yb_ = yb.view(-1)
        
        total_loss   += loss.item() * xb.size(0)
        total       += xb.size(0)
        true_y.extend(list(yb_.detach().cpu()))
        pred_y.extend(list(logits_.detach().cpu()))
        
    pcc,_ = sp.stats.pearsonr(true_y,pred_y)
    mse = mean_squared_error(true_y, pred_y)
    return total_loss/total,pcc,pow(mse,0.5),model

def eval_model(model, dl, device,year,seed,e,epoch):
    model.eval()
    true_y,pred_y = [],[]
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_ = logits.view(-1)
            yb_ = yb.view(-1)
            true_y.extend(list(yb_.detach().cpu().numpy()))
            pred_y.extend(list(logits_.detach().cpu().numpy()))
    pcc,_ = sp.stats.pearsonr(true_y,pred_y)
    mse = mean_squared_error(true_y, pred_y)
    return pcc,pow(mse,0.5)

def save_prediction(model, dl, device, year, seed, e, epoch,folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
            
    model.eval()
    true_y,pred_y = [],[]
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_ = logits.view(-1)
            yb_ = yb.view(-1)
            true_y.extend(list(yb_.detach().cpu().numpy()))
            pred_y.extend(list(logits_.detach().cpu().numpy()))
    if seed==1:
        np.save(folder+'/'+str(year)+'-true.npy',true_y)
    np.save(folder+'/'+str(year)+'-seed-'+str(seed)+'-pred.npy',pred_y)


def get_metrics(year,seeds,folder):
    true_y = np.load(folder + '/'+str(year)+'-true.npy')
    pred = []
    
    for seed in seeds:
        pred_y = np.load(folder + '/'+str(year)+'-seed-'+str(seed)+'-pred.npy')
        pred.append(pred_y)

    
    '''
    for seed in seeds:
        pred_y = np.load('../final-code/charge-gbt-prediction/'+str(year)+'-seed-'+str(seed)+'-pred.npy')
        pred.append(pred_y)
    
    pcc = []
    for pred_y in pred:
        pcc1,_ = sp.stats.pearsonr(true_y,pred_y)
        print(pcc1)
        pcc.append(pcc1)
    pcc_mean = np.mean(pcc)
    print(f'Average PCC: {pcc_mean:.3f}')
    '''
    
    # compute
    pred = np.array(pred)
    pred_mean = pred.mean(axis=0)
    pcc,_ = sp.stats.pearsonr(true_y,pred_mean)
    mse = mean_squared_error(true_y, pred_mean)
    rmse = pow(mse,0.5)
    print(f'Final PCC: {pcc:.3f}, RMSE: {rmse:.3f}')


