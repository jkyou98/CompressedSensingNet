
from pyexpat import model
from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
import datetime


def fuzzy_loss(losses, m):
    total_loss = 0.0
    mus = []
    for loss1 in losses:
        s = 0.0
        for loss2 in losses:
            s += (loss1 / loss2)
        mu = (1 / s) ** (1 / (m - 1))
        mus.append(mu)
        total_loss += mu * loss1
    return total_loss, mus

def training_loop(device, n_epochs, optimizer, model, train_loader, val_loader, fuzzyMTLloss = True, fuzzy_m=2):
#     torch.autograd.set_detect_anomaly(True)
    model.to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    loss_list=[]
    loss1_list=[]
    loss2_list=[]
    loss3_list=[]
    val_loss_list=[]
    mus_list=[]
    val_mus_list=[]

    for epoch in range(1, n_epochs + 1): 
        loss_train = 0.0
        loss1_train =  0.0
        loss2_train =  0.0
        loss3_train =  0.0

        val_loss_train = 0.0

        model.train()
        for imgs in train_loader:
            imgs=imgs.to(device)
            outputs,thetaZ,Z_hat,Y,Z = model(imgs)
#             loss =  nn.MSELoss()(outputs, imgs)+nn.MSELoss()(torch.norm(compress_output),torch.norm(imgs))
            #loss = nn.MSELoss()(torch.norm(compress_output),torch.norm(imgs))
            loss1 = nn.MSELoss(reduction='none')(outputs, imgs)
            loss1 = torch.sum(loss1)/(outputs.size(1)*outputs.size(2)*outputs.size(3))
            loss2 = nn.MSELoss(reduction='none')(Z_hat, imgs)
            loss2 = torch.sum(loss2)/(Z_hat.size(1)*Z_hat.size(2)*Z_hat.size(3))
            loss3 =Z.abs().sum()

            if fuzzyMTLloss:
                loss, mus = fuzzy_loss([loss1, loss2, loss3], fuzzy_m)
            else:
                loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
#             with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            loss1_train += loss1.item()
            loss2_train += loss2.item()
            loss3_train += loss3.item()

        loss_train = loss_train/len(train_loader.sampler)
        loss1_train = loss1_train/len(train_loader.sampler)
        loss2_train = loss2_train/len(train_loader.sampler)
        loss3_train = loss3_train/len(train_loader.sampler)        
   
        
        loss_list.append(loss_train)
        loss1_list.append(loss1_train)
        loss2_list.append(loss2_train)
        loss3_list.append(loss3_train)
        mus_list.append(mus)

        model.eval()
        with torch.no_grad():
            for val_imgs in val_loader:
                val_imgs=val_imgs.to(device)
                val_outputs,val_thetaZ,val_Z_hat,val_Y,val_Z = model(val_imgs)
                val_loss1 = nn.MSELoss(reduction='none')(val_outputs, val_imgs)
                val_loss1=torch.sum(val_loss1)/(val_outputs.size(1)*val_outputs.size(2)*val_outputs.size(3))
                val_loss2 = nn.MSELoss(reduction='none')(val_Z_hat, val_imgs)
                val_loss2 = torch.sum(val_loss2)/(val_Z_hat.size(1)*val_Z_hat.size(2)*val_Z_hat.size(3))
                val_loss3 =val_Z.abs().sum()

                if fuzzyMTLloss:
                    val_loss, val_mus = fuzzy_loss([val_loss1, val_loss2, val_loss3], fuzzy_m)
                else:
                    val_loss = val_loss1 + val_loss2 + val_loss3
                val_loss_train += val_loss.item()

            val_loss_train = val_loss_train/len(val_loader.sampler)
        scheduler.step(val_loss_train)     

        val_loss_list.append(val_loss_train)
        val_mus_list.append(val_mus)
        if epoch == 1 or epoch % 1 == 0:
            print('{} Epoch {}, Training loss {}, Validation loss {}'.format (
                datetime.datetime.now(), epoch, float(loss_train), float(val_loss_train)))

    return  loss_list,loss1_list,loss2_list,loss3_list, val_loss_list, mus_list, val_mus_list