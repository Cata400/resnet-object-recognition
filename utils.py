import os
import time
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, test_loader, optimizer, criterion, epochs, device, scheduler, save_model_name):
    start = time.time()
    test_accuracies = []
    writer = SummaryWriter(log_dir='Logs')
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        
        model.train()
        train_iter_loss = 0.0
        train_iter_acc = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            
            train_iter_loss += loss.item() / len(train_loader)
            train_iter_acc += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item() / len(train_loader)
        
        scheduler.step()
        print(f'Training loss: {train_iter_loss:.4f}\t Training accuracy: {train_iter_acc:.4f}')
        writer.add_scalar('Loss/train', train_iter_loss, epoch)
        writer.add_scalar('Accuracy/train', train_iter_acc, epoch)
        
        model.eval()
        test_iter_loss = 0.0
        test_iter_acc = 0
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
                        
            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)
            
            test_iter_loss += loss.item() / len(test_loader)
            test_iter_acc += (y_hat.argmax(1) == y.argmax(1)).type(torch.float).sum().item() / len(test_loader)
            
        print(f'Test loss: {test_iter_loss:.4f}\t Test accuracy: {test_iter_acc:.4f}')
        writer.add_scalar('Loss/test', test_iter_loss, epoch)
        writer.add_scalar('Accuracy/test', test_iter_acc, epoch)
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_model_name + f'_{epoch + 1}.pth')
            test_accuracies.append(test_iter_acc)
        
        print()
        
    stop = time.time()
    print(f'Training time: {stop - start:.2f} seconds')
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