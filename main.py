import os
import torch

import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import Caltech256
from torchvision.models import resnet18, resnet50, resnet101, wide_resnet101_2, resnext101_64x4d, \
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, Wide_ResNet101_2_Weights, ResNeXt101_64X4D_Weights
from utils import *


if __name__ == '__main__':
    # Script parameters
    dataset_path = 'Caltech_256'
    classes_path = os.path.join(dataset_path, 'caltech256', '256_ObjectCategories')
    classes = sorted(os.listdir(classes_path))
    no_classes = len(classes)
    
    save_model_name = 'resnet18_trained_freeze_conv'
    
    results_name = 'results_freeze_conv'
    results_path = os.path.join('Results', results_name + '.csv')
    
    logdir = os.path.join('Logs', 'log_' + save_model_name.split('.')[0] + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 50
    split = 0.9
    
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    # Load and preprocess data
    # grayscale_imgs_count = check_dataset_shapes(classes_path)
    # print(f'Grayscale images count: {grayscale_imgs_count}')
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        Gray2RGB(),
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
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    # Load model
    # model = MLP6(3 * 224 * 224, no_classes)
    model = resnet18(weights=None)
    
    # Freeze convolutional layers
    # for param in model.parameters():
    #     param.requires_grad = False
        
    in_features_fc = model.fc.in_features
    model.fc = torch.nn.Linear(in_features_fc, no_classes)
    model = model.to(device)
    print(f'Model number of parameters: {count_parameters(model)}', flush=True)
    
    # Train model, save model every 10 epochs, monitor it using tensorboard, keep track of computation time
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model, test_accuracies_1, training_time_1 = train(model, train_dataloader, test_loader, optimizer, criterion, 0, epochs, device, lr_scheduler, save_model_name, logdir)
    
    # Unfreeze convolutional layers
    # for param in model.parameters():
    #     param.requires_grad = True
                
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # model, test_accuracies_2, training_time_2 = train(model, train_dataloader, test_loader, optimizer, criterion, epochs - 20, epochs, device, lr_scheduler, save_model_name, logdir)
    
    test_accuracies_2 = []
    training_time_2 = 0
    
    test_accuracies = test_accuracies_1 + test_accuracies_2
    training_time = training_time_1 + training_time_2
    
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
    # results_df = results_df.append(row, ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    
    # Get confusion matrix
    
    
    
