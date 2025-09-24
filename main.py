from pathlib import Path
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import BSDS, BSDS_val
from src.model import CompressedSensingNet 
from src.train import training_loop
from src.data_argumentation import run_data_augmentation
from src.test import test_loop



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Use Device: {device}")
    
    #prepare BDSD500 dataset
    
    rootDirImgTrain = "BSDS500/data/images/train/"
    destDirImgTrain = "BSDS500_AUGMENTED/data/images/train/"
    rootDirImgVal = "BSDS500/data/images/val/"
    rootDirImgTest = "BSDS500/data/images/test/"

    pairs = [
        (rootDirImgTrain, destDirImgTrain)
    ]
    augmented_root = Path("BSDS500_AUGMENTED")
    
    if augmented_root.exists():
        print(f"Augmented dataset already exists at {augmented_root}. Skipping augmentation.")
    else:
        summary = run_data_augmentation(
            source_dest_pairs=pairs,
            new_size=(400, 400),
            num_angles=16,   #angles=[0, 10, 20, ...]
            verbose=True
        )
        print(summary)
        
    #import dataset to tensor    
    trainDS=BSDS(destDirImgTrain)
    valDS=BSDS_val(rootDirImgVal)
    testDS=BSDS_val(rootDirImgTest)
    
    model = CompressedSensingNet()
    
    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=64, shuffle=True,num_workers=0,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valDS, batch_size=64, shuffle=False,num_workers=0,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testDS, batch_size=1, shuffle=False,num_workers=0,pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-07) #lr=5e-8


    loss_list,loss1_list,loss2_list,loss3_list, val_loss_list, mus_list, val_mus_list=training_loop(
        device = device,
        n_epochs = 50,
        optimizer = optimizer,
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        fuzzy_m = 2
        )
    
    psnr = test_loop(
        device = device,
        model = model,
        test_loader = test_loader
    )
    
    print(f"Test PSNR: {psnr:.2f} dB")

    # Total loss
    plt.figure()
    plt.plot(loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Total Loss")
    plt.savefig("total_loss.png")

    # Loss1
    plt.figure()
    plt.plot(loss1_list, label="Loss1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss1")
    plt.legend()
    plt.title("Loss1 Curve")
    plt.savefig("loss1_curve.png")

    # Loss2
    plt.figure()
    plt.plot(loss2_list, label="Loss2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss2")
    plt.legend()
    plt.title("Loss2 Curve")
    plt.savefig("loss2_curve.png")

    # Loss3
    plt.figure()
    plt.plot(loss3_list, label="Loss3")
    plt.xlabel("Epoch")
    plt.ylabel("Loss3")
    plt.legend()
    plt.title("Loss3 Curve")
    plt.savefig("loss3_curve.png")
    
    
    
    

