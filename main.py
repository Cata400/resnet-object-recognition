import os
import torch

import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import Caltech256
from torchvision.models import resnet18, resnet50, resnet101, wide_resnet101_2, resnext101_64x4d
from utils import *


if __name__ == '__main__':
    # Script parameters
    dataset_path = 'Caltech_256'
    classes_path = os.path.join(dataset_path, 'caltech256', '256_ObjectCategories')
    classes = sorted(os.listdir(classes_path))
    no_classes = len(classes)
    
    save_model_name = 'resnet18_finetuned_full'
    model_path = os.path.join('Models', save_model_name + '.pth')
    
    results_name = 'results_finetuned_full'
    results_path = os.path.join('Results', results_name + '.csv')

    # Hyperparameters
    batch_size = 4
    lr = 1e-3
    epochs = 50
    split = 0.7
    
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    # Load and preprocess data
    # grayscale_imgs_count = check_dataset_shapes(classes_path) # TODO: fix grayscale images
    # print(f'Grayscale images count: {grayscale_imgs_count}')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(), 
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
    model = resnet18(pretrained=True)
    in_features_fc = model.fc.in_features
    model.fc = torch.nn.Linear(in_features_fc, no_classes)
    # model = torch.compile(model)
    model.to(device)
    print(f'Model number of parameters: {count_parameters(model)}')
    
    # Train model, save model every 10 epochs, monitor it using tensorboard, keep track of computation time
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    model, test_accuracies, training_time = train(model, train_dataloader, test_loader, optimizer, criterion, epochs, device, lr_scheduler, save_model_name)
    
    # Write the accuracies and time into a csv file
    results_df = pd.read_csv(results_path)
    row = {
            'Model': save_model_name, 
            'Split': f"{int(split * 100)} - {int((1 - split) * 100)}", 
            'Batch_size': batch_size,
            'Initial_lr': lr,
            'Accuracy_epoch_10': test_accuracies[0],
            'Accuracy_epoch_20': test_accuracies[1],
            'Accuracy_epoch_30': test_accuracies[2],
            'Accuracy_epoch_40': test_accuracies[3],
            'Accuracy_epoch_50': test_accuracies[4],
            'Time': training_time
        }
    results_df = results_df.append(row, ignore_index=True)
    results_df.to_csv(results_path, index=False)
    
    # Get confusion matrix
    
    
    
