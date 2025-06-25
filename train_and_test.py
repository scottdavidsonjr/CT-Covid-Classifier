from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as models
from skimage.util import montage
import os 
import cv2
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg19_bn
import random
import functions_and_classes as f
import time
import subprocess
import webbrowser
from sklearn.model_selection import train_test_split

# Log start time of code execution
start_time = time.perf_counter()

#  ----------------------------------------- TENSORBOARD ACTIVATION ----------------------------------------- #
# To log information in tensorboard
log_dir = '/Users/scottdavidson/Desktop/Research/CNN_Covid_Project/logs_Resnet'
writer = SummaryWriter(log_dir)

# Automatically start TensorBoard in the background
subprocess.Popen(['tensorboard', '--logdir', log_dir])
# Wait a few seconds to ensure TensorBoard has started
time.sleep(3)
# Open TensorBoard in the default web browser (port 6006 is the default TensorBoard port)
webbrowser.open("http://localhost:6006")


#  ----------------------------------------- IMPORT DOWNLOADED DATA ----------------------------------------- #
img_dir_path =  'Patients/' #'/Users/scottdavidson/Desktop/Research/CNN_Covid_Project/'
covid_img_path = img_dir_path + 'YES_COVID/2A_images/2A_images/'
non_covid_img_path = img_dir_path + 'NON_COVID/2A_images/2A_images/'

## Read In & Plot Non-COVID Images
non_covid_files = [os.path.join(non_covid_img_path, x) for x in os.listdir(non_covid_img_path)] # Gets each image and retrieves its full path
non_covid_images = [cv2.imread(x) for x in random.sample(non_covid_files, 15)]
f.show_sample_images(non_covid_images) # View sample non-covid images

## Read In & Plot COVID Images
covid_files = [os.path.join(covid_img_path, x) for x in os.listdir(covid_img_path)] # Gets each image and retrieves its full path
covid_images = [cv2.imread(x) for x in random.sample(covid_files, 15)]
f.show_sample_images(covid_images) # view sample covid images

# Remove '.DS_Store' files- ERROR WHEN PROCESSING FILES WHICH NEEDS REMOVAL
non_covid_files = [file for file in non_covid_files if '.DS_Store' not in file]
covid_files = [file for file in covid_files if '.DS_Store' not in file]

image_files = covid_files + non_covid_files # combine data

#  ----------------------------------------- SPLITTING UP DATA ----------------------------------------- #

# Get ratios (adjust as needed)
train_ratio = 0.85
val_ratio = 0.075
test_ratio = 0.075

train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio), random_state=17)
val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=17)

# Determine covid & non-covid splits & total sizes for each:
for lst in [train_files, val_files, test_files]:
    tot = len(lst)
    non_cov = sum('NON_COVID' in file for file in lst)
    cov = tot - non_cov
    
    print('Covid:', cov, 'Non-Covid:', non_cov, 'Total:', tot , '\n')

#  ----------------------------------------- AUGMENTING DATA ----------------------------------------- #

# For normalization, use ImageNet normalization values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Apply transformations
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomResizedCrop((224),scale=(0.9,1.1)), # Crops image randomly and resizes to 224 x 224 while zooming in or out within given range
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Slightly shifts images at random
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1,2.0)), # Blurs pixels of the image
    transforms.Grayscale(num_output_channels=3), # Make it grayscale; Need 3 channels for Resnet
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3), # Make it grayscale; Need 3 channels for Resnet
    transforms.ToTensor(),
    normalize
])


batchsize = 128 # Adjust as desired- this is what my GPU could handle with Resnet model

trainset = f.CovidCTDataset(image_files = train_files, 
                          transform= train_transformer)

valset = f.CovidCTDataset(image_files = val_files, 
                          transform= val_transformer)

testset = f.CovidCTDataset(image_files = test_files, 
                          transform= val_transformer)

#  ----------------------------------------- LABELING DATA ----------------------------------------- #
# Send labels to GPU
labels = [elem['label'] for elem in trainset]
labels = f.torch.tensor(labels)  

### Account for class imbalances ###

# Get training label counts
train_non_covid = sum('NON_COVID' in file for file in train_files)
train_covid = len(train_files) - train_non_covid
class_sample_counts = [train_covid, train_non_covid]


#  ----------------------------------------- UPSAMPLING, PREPPING DATA & MODEL ----------------------------------------- #
# Minority class gets higher weight to encourage model to see non-covid cases (minority class) more often
weights = 1. / f.torch.tensor(class_sample_counts, dtype=f.torch.float) 
sample_weights = weights[labels]  # labels is a tensor of your dataset's labels
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(trainset, sampler= sampler,  batch_size=batchsize, drop_last=False,  num_workers=0, pin_memory=False) # Can't use shuffle & sampler at same time; shuffle=True,
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=0, pin_memory=False)

### Resnet18
model = models.resnet18(weights='DEFAULT')
model.fc = f.nn.Linear(model.fc.in_features, 2) # Model fully connected layers
model = model.to(f.device)

### Hyperparameter tuning
learning_rate = 0.01 # Standard learning rate; don't want convergence to be too fast and overshoot local/absolute minimum for optimization
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9) # Momentum helps to overcome local minima



#  ----------------------------------------- TRAINING MODEL ----------------------------------------- #

# best_model = model
best_val_score = 99999999999 # Loss instead of accuracy

class_weights = f.torch.tensor([1.0, 1107 / 433], dtype=f.torch.float).to(f.device)  # [Covid, Non-Covid]
criterion = f.nn.CrossEntropyLoss(weight=class_weights) # This is the loss function, AKA log loss

early_stopper = f.EarlyStopping(patience = 5) # Set this up for early stopping

# Check that model & tensors are on mps
print(next(model.parameters()).device)  # should be mps:0

for epoch in range(60):
    if epoch > 9:
        break

    model.train()
    train_loss=0
    train_correct=0


    for iter_num, data in enumerate(train_loader): # Data is clumped into batches of 16, since that is what was specified in DataLoader class
 

        image, target = data['img'].to(f.device), data['label'].to(f.device)

        # Compute the loss
        output = model(image.float()) # Each neuron is -1 to 1 for each of 8 images
            # Do float to ensure images are float32, not float64
        loss = criterion(output, target)

        # Scale loss for accumulation

        loss.backward() # Performs backpropagation to compute the gradient of the loss for each parameter
        optimizer.step() # Update model parameters using gradients
        optimizer.zero_grad() # Clear gradients from previous step

        # Log loss, scale back up to original scale
        train_loss += loss.item()
        
        f.torch.mps.synchronize() # Force sync to help GPU activity show up 
        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True) # Prediction for each image
        train_correct += pred.eq(target.view_as(pred)).sum().item() # Number predicted correctly
        

    # Compute & print performance metrics:
    print("Val files:", len(val_files))

    metrics_dict = f.compute_metrics(model, val_loader)

    print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,
                                                                                                 iter_num))
    print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
    print("------------------------------------------------------------------------------")
    
    
    # Save model with best validation accuracy
    if metrics_dict['Validation Loss'] < best_val_score:
        f.torch.save(model, "best_model_Resnet.pkl") # torch is imported in f, so utilize f module
            # Above saves serialized model (serialized means converting object into series of bytes that is saved in an easily transmittable state)
            # Pickle module- implements binary protocols for serializing and unserializing data easily
        best_val_score = metrics_dict['Validation Loss']


    # print the metrics for training data for the epoch
    print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    
     # log the accuracy and losses in tensorboard
    writer.add_scalars( "Losses", {'Train loss': train_loss / len(train_loader), 'Validation_loss': metrics_dict["Validation Loss"]},
                                   epoch)
    writer.add_scalars( "Accuracies", {"Train Accuracy": 100.0 * train_correct / len(train_loader.dataset),
                                       "Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

    if early_stopper.stop() == 3:
        for param_group in optimizer.param_groups:
            learning_rate *= 0.1
            param_group['lr'] = learning_rate
            print('Updating the learning rate to {}'.format(learning_rate))
            early_stopper.reset()



writer.close()


# Determine time it took to train the model
end_time = time.perf_counter()
execution_time = (end_time - start_time) / 60
print(f"Execution time: {execution_time:.4f} minutes")


### TESTING MODEL PERFORMANCE ###
model_trained = f.torch.load("best_model_Resnet.pkl", weights_only=False)


metrics_dict = f.compute_metrics(model_trained, test_loader, plot_roc_curve = True)
print('------------------- Test Performance --------------------------------------')
print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
print("------------------------------------------------------------------------------")


