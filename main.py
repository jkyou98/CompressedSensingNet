from pathlib import Path
import torch
from src.data_argumentation import run_data_augmentation
from torch.utils.data import DataLoader
import numpy as np

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
    rootDirImgVal   = "BSDS500/data/images/val/"

    destDirImgTrain = "BSDS500_AUGMENTED/data/images/train/"
    destDirImgVal   = "BSDS500_AUGMENTED/data/images/val/"

    pairs = [
        (rootDirImgTrain, destDirImgTrain),
        (rootDirImgVal,   destDirImgVal)
    ]
    augmented_root = Path("BSDS500_AUGMENTED")
    
    if augmented_root.exists():
        print(f"Augmented dataset already exists at {augmented_root}. Skipping augmentation.")
    else:
        summary = run_data_augmentation(
            source_dest_pairs=pairs,
            new_size=(400, 400),
            num_angles=16,   # 或 angles=[0, 10, 20, ...]
            verbose=True
        )
        print(summary)
    
    # model = Net1().to(device)
    # train_dataset, test_dataset = get_cifar10_datasets()
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32)

    # training_loop(model, train_loader, device)
    # validate(model, test_loader, device)
