import matplotlib.pyplot as plt 
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from collections import deque
import numpy as np

# SPECIFIC TO USER #
# FOR MAC, USE METAL PERFORMANCE SHADERS FOR GPU ACCELERATION#
print(torch.backends.mps.is_available())
device = torch.device('mps')


### IMAGE DEMONSTRATION -----------------------------------------
    # For .png files #
def show_sample_images(images):
    plt.figure(figsize=(20,10))
    columns = 5

    for i, image in enumerate(images):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        arr = np.asarray(image)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
        plt.axis('off') # Turn off axis for each plot

    plt.tight_layout() # Adjust layout so plots don't overlap
    plt.show()



### DATASET CLASS--------------------------------------------
class CovidCTDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_list = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]

        # Assign labels to images:
        if ('YES_COVID' in path):
            label = 1
        else:
            label = 0
        
      # Read the image
        image = Image.open(path) #.convert('RGB')
    
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data



### METRIC COMPUTATION---------------------------------------------------
def compute_metrics(model, test_loader, plot_roc_curve = False):
    
    model.eval()
    
    val_loss = 0
    val_correct = 0
    
    criterion = nn.CrossEntropyLoss()
    
    score_list   = torch.Tensor([]).to(device)
    pred_list    = torch.Tensor([]).to(device)
    target_list  = torch.Tensor([]).to(device)
    path_list    = []

    
    for iter_num, data in enumerate(test_loader):
        # if (iter_num > 4):
        #     break
        # Convert image data into single channel data
        image, target = data['img'].to(device), data['label'].to(device)
        paths = data['paths']
        path_list.extend(paths) # Extend adds all elements in paths to end of path_list
        
        # Compute the loss
        with torch.no_grad():
            output = model(image)
        
        # Log loss
        val_loss += criterion(output, target).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Bookkeeping 
        score_list   = torch.cat([score_list, nn.Softmax(dim = 1)(output)[:,1].squeeze()])
        pred_list    = torch.cat([pred_list, pred.squeeze()]) # Remove all dimensions with size of 1 from the array
        target_list  = torch.cat([target_list, target.squeeze()])
        
        # print(pred, val_correct)
        # print('\n', score_list, pred_list, target_list)
    
    classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(),
                                                labels = [0, 1],
                                                target_names = ["Non-COVID", "COVID"], # Optional display names matching the labels (same order)
                                                output_dict= True)
    
    
    # sensitivity is the recall of the positive class (COVID)
    sensitivity = classification_metrics['COVID']['recall']
    
    # specificity is the recall of the negative class (Non-COVID)
    specificity = classification_metrics['Non-COVID']['recall']

    # print(classification_metrics)
    
    # accuracy
    accuracy = classification_metrics['accuracy']
    
    # confusion matrix
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())

    TP = conf_matrix[1,1]
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]

    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    
    # roc score
    roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())
    
    # plot the roc curve if parameter set to true
    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
        plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
        plt.legend(loc = 'best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        
    
    # put together values
    metrics_dict = {"Accuracy": accuracy,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Roc_score"  : roc_score, 
                    "Confusion Matrix": conf_matrix,
                    "PPV": PPV,
                    "NPV": NPV,
                    "Validation Loss": val_loss / len(test_loader),
                    "score_list":  score_list.tolist(),
                    "pred_list": pred_list.tolist(),
                    "target_list": target_list.tolist()} #,
                    # "paths": path_list}
    
    
    return metrics_dict



### Early Stopping----------------------------------

# New Early Stopping Class
class EarlyStopping(object):
    def __init__(self, patience = 5, min_delta = 0): # Can be zero since val changes are not sudden according to prev training sessions
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = int(1e7)
        self.loss_iter = 0

    def add_data(self, loss):
        if loss < (self.best_loss - self.min_delta): # Lower means loss is improving; Want to have option of min_delta
            self.best_loss = loss
        else:
            self.loss_iter+= 1
            print(self.loss_iter)

    def stop(self):
        bool_val = False
        if self.loss_iter >= self.patience:
            bool_val = True
            
        return(bool_val)