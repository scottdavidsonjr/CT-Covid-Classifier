# CT-Covid-Classifier
Utiliizing ResNet-18 (pre-trained on ImageNet dataset, fine tuned on 100k> images) to classify Covid-19 from Chest CT images

**Instructions**
1. Run get_data.py
  a. Download data from relevant Kaggle site
  b. Get path of where downloaded data was stored and adjust code where necessary
2. Run train_and_test in conjunction with functions_and_classes to train model (store weights for Resnet-18) then test performance
  a. On Mac using Metal Performance Shaders (mps), it should take about 2.5 hours.

**Results:**
- Accuracy: 96.7%
- PPV (Precision): 95.8%
- Sensitivity (Recall): 95.2%
