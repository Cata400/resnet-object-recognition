import os
import torch

import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import Caltech256
from torchvision.models import resnet18, resnet50, resnet101, wide_resnet101_2, resnext101_64x4d, \
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNeXt101_64X4D_Weights, Wide_ResNet101_2_Weights

from utils import *


if __name__ == '__main__':
    # Script parameters
    dataset_path = os.path.join('..', 'Caltech_256')
    classes = sorted(os.listdir(os.path.join(dataset_path, 'caltech256', '256_ObjectCategories')))
    no_classes = len(classes)
    
    save_model_name = 'model.h5'

    # Hyperparameters
    batch_size = 32
    lr = 1e-3
    epochs = 50
    split = 0.7
    
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    # Load and preprocess data
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), 
        # transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset = Caltech256(root=dataset_path, 
                        transform=transform, 
                        target_transform=transforms.Lambda(lambda y: torch.zeros(no_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)), 
                        download=False)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(split * len(dataset)), len(dataset) - int(split * len(dataset))])
    
    training_set_size = len(train_set)
    test_set_size = len(test_set)
    
    print(f'Training set size: {training_set_size}')
    print(f'Test set size: {test_set_size}')
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    
    # Load model
    
    
    # Train model using transfer learning, save model every 10 epochs and monitor it using tensorboard
    
    
    # Load best model and evaluate it on test set
    
    
    # Get confusion matrix
    
    
    
