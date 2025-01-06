import sklearn as sk
from sklearn.metrics import accuracy_score
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss as NBDL
import numpy as np
from torch import nn
import loss as l
import copy
import torch
import torchvision as tv
from anndata.experimental import AnnLoader
from typing import Callable, List, Optional, Sequence, Tuple, Union

def val_loss(model, dataloader, criterion,label_type, encoder):
  val_loss = 0.0
  print(len(dataloader))
  with torch.no_grad():
    for i,batch in enumerate(dataloader):
        if label_type == "cell_type":
            if type(dataloader) is AnnLoader:
                feature = batch.obsm['X_pca'].float()
                label = torch.tensor(model.encoder.transform(batch.obs['supercluster_term']))
            else:
                feature,label = batch
            if encoder:
                feature = encoder.encode(feature)
        elif label_type == "auto_enc":
            if type(dataloader) is AnnLoader:
                feature = batch.X.float()
            else:
                feature = batch
            label = feature
        feature = feature.to('cuda')
        label = label.to('cuda')
        pred = model(feature)
        loss = criterion(pred,label)
        print(loss.item())
        val_loss += loss.item()/len(dataloader)
    return val_loss

def train_val(model, train_loader, val_loader,epochs,loss_type="cell_type",start_epoch=0, encoder = None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=model.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, start_epoch+1)
    criterion = None
    if loss_type == "cell_type":
        criterion = nn.NLLLoss()
        label_type = "cell_type"
    elif(loss_type == "MSE"):
        criterion = nn.MSELoss()
        label_type = "auto_enc"
    elif(loss_type == "NBDL"):
        criterion = NBDL() 
        label_type = "auto_enc"
    elif(loss_type=="ZINB"):
        zinb = l.ZINB()
        criterion = lambda pred,label: zinb.loss(pred,label).sum(dim=1).mean()
        label_type = "auto_enc"
    else:
        raise Exception("invalid loss type")
    for epoch in range(start_epoch,epochs):
      model.train()
      loss = train_epoch(dataloader=train_loader,model=model,optimizer=optimizer,criterion=criterion, label_type=label_type, epoch_num=epoch, encoder = encoder,lr_scheduler = lr_scheduler)
      print('Epoch:', epoch,', Loss:', loss)
      model.eval()
      val = val_loss(dataloader=val_loader, model = model, criterion = criterion,label_type=label_type, encoder = encoder)
      
    PATH = "regular_linear"
    torch.save(model.state_dict(), PATH)
    print('Validation loss', val)




def train(dataloader,model,epochs,loss_type="cell_type",start_epoch=0, encoder = None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=model.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, start_epoch+1)
    if loss_type == "cell_type":
        criterion = nn.NLLLoss()
        label_type = "cell_type"
    elif(loss_type == "MSE"):
        criterion = nn.MSELoss()
        label_type = "auto_enc"
    elif(loss_type == "NBDL"):
        criterion = NBDL() 
        label_type = "auto_enc"
    elif(loss_type=="ZINB"):
        zinb = l.ZINB()
        criterion = lambda pred,label: zinb.loss(pred,label)
        label_type = "auto_enc"
    else:
        raise Exception("invalid loss type")
    for epoch in range(start_epoch,epochs):
        loss = train_epoch(dataloader=dataloader,model=model,optimizer=optimizer,criterion=criterion, label_type=label_type, epoch_num=epoch, encoder = encoder,lr_scheduler = lr_scheduler)
        print('Epoch:', epoch,', Loss:', loss)
        #PATH = "drive/MyDrive/CSE527/large_trained_auto_transformer_ZINB"
        #PATH = "drive/MyDrive/CSE527/basic_encoder_mse"
        #PATH = "drive/MyDrive/CSE527/at_wa_mha_patch"
        #PATH = "drive/MyDrive/CSE527/scvi_classifier"
        #torch.save(model.state_dict(), PATH)
    model.eval()


def train_epoch(dataloader,model,optimizer,criterion,lr_scheduler, epoch_num,batches=-1,label_type="cell_type", encoder = None):
    batch_count = 0
    avg_loss = 0
    for i,batch in enumerate(dataloader):
        if label_type == "cell_type":
            if type(dataloader) is AnnLoader:
                if encoder == None:
                    if 'X_scvi' in batch.obsm.keys():
                        feature = batch.obsm['X_scvi'].float().to('cuda')
                    if 'X_pca' in batch.obsm.keys():
                        feature = batch.obsm['X_pca'].float().to('cuda')
                else: 
                  feature = batch.X.float()
                feature = feature.to('cuda')
                label = torch.tensor(model.encoder.transform(batch.obs['supercluster_term'])).to('cuda')
            else:
                feature,label = batch
            if encoder:
                feature = encoder.encode(feature)
        elif label_type == "auto_enc":
            if type(dataloader) is AnnLoader:
                feature = batch.X.float().to('cuda')
            else:
                feature = batch
            label = feature
        pred = model(feature)
        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch_num+i/len(dataloader))
        if batch_count == batches:
            break
        batch_count += 1
        avg_loss += loss.item()/len(dataloader)
        print("Batch ", batch_count, " Loss ", loss.item())
    return avg_loss