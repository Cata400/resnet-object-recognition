import datetime
import os
import time
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter


class Gray2RGB(object):
    def __call__(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        return image


def train(model, train_loader, test_loader, optimizer, criterion, epochs_low, epochs_high, device, scheduler, save_model_name, logdir):
    start = time.time()
    test_accuracies = []
    writer = SummaryWriter(log_dir=logdir)
    
    training_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    for epoch in range(epochs_low, epochs_high):
        print(f'Epoch {epoch + 1}/{epochs_high}')
        print('-' * 10)
        
        model.train()
        train_iter_loss = torch.zeros(1, device=device)
        train_iter_acc = torch.zeros(1, device=device)
        
        for i, (x, y) in enumerate(train_loader):
            # print(f'Iteration {i + 1}/{len(train_loader)}')
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            
            train_iter_loss += loss / training_size
            train_iter_acc += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum() / training_size
        
        scheduler.step()
        print(f'Training loss: {train_iter_loss.item():.4f}\t Training accuracy: {train_iter_acc.item():.4f}', flush=True)
        
        model.eval()
        test_iter_loss = 0.0
        test_iter_acc = 0
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
                        
            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)
            
            test_iter_loss += loss / test_size
            test_iter_acc += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum() / test_size
            
        print(f'Test loss: {test_iter_loss.item():.4f}\t Test accuracy: {test_iter_acc.item():.4f}', flush=True)
        
        writer.add_scalars('Loss', {'train': train_iter_loss, 'test': test_iter_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_iter_acc, 'test': test_iter_acc}, epoch)
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join('Models', save_model_name + f'_{epoch + 1}.pth'))
            test_accuracies.append(test_iter_acc.item())
        
        print()
        
    stop = time.time()
    print(f'Training time: {stop - start:.2f} seconds\n')
    writer.flush()
    writer.close()
    
    return model, test_accuracies, stop - start


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_dataset_shapes(dataset_path):
    count = 0
    for cls in os.listdir(dataset_path):
        for img in os.listdir(os.path.join(dataset_path, cls)):
            img_path = os.path.join(dataset_path, cls, img)
            img = torchvision.io.read_image(img_path)
            
            if img.shape[0] != 3:
                print(f'{img_path} has {img.shape[0]} channels')
                count += 1
                
    return count
                

def get_confusion_matrix(y_true, y_pred, labels, plot_cm=True, scale=10, print_acc=False):
    """
    Get confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list or tuple of shape (n_samples,) 
        List of labels.
    plot_cm : bool, optional
        True if the confusion matrix is to be plotted, False otherwise, by default True.
    scale : int, optional
        Scale of the plotted confusion matrix for better reading, by default 10.
    print_acc : bool, optional
        True to print the per-class accuracy for every class, False otherwise, by default False.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    if plot_cm:
        cm_cut = cm[:scale, :scale]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_cut, display_labels=labels[:scale])
        disp.plot()
        plt.show()
    
    if print_acc:
        for i, cls in enumerate(cm):
            print(f'{labels[i]}: {np.round(100 * cls[i],2 )}% accuracy' )
            
            

class MLP6(torch.nn.Module):
    """
    Multilayer perceptron with 6 hidden layers.

    Attributes
    ----------
    input_size : int
        Size of input.
    no_classes : int
        Number of classes.
    flatten : torch.nn.Flatten
        Flatten layer.
    linear_relu_stack : torch.nn.Sequential
        Sequential layer.
    softmax : torch.nn.Softmax
        Softmax layer.
    """
    def __init__(self, input_size, no_classes):
        super(MLP6, self).__init__()
        self.input_size = input_size
        self.no_classes = no_classes
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.no_classes),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        
        return y
