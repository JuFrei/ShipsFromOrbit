# Classifying Ships from Orbit
This python notebook contains code for training a Convolutional Neural Network (CNN) using PyTorch for image classification tasks. Additionally, the code includes functionality for training a Support Vector Machine (SVM) classifier using Histogram of Oriented Gradient (HOG) features for comparison.

## Data
### Source
https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery

### Description
The dataset comprises 4000 RGB images, each sized 80x80 pixels, extracted from Planet satellite imagery covering the San Francisco Bay and San Pedro Bay regions of California. These images are classified as either "ship" or "no-ship".

### Labels
The dataset comprises two classes:

* **"Ship" Class**: This class consists of 1000 images, each centered on the body of a single ship.

* **"No-Ship" Class**: This class comprises 3000 images. A third of the images representing various land cover features such as water bodies, vegetation, bare earth, and buildings, without any portion of a ship. Another third containing "partial ships" that show only a portion of a ship. The final third consists of images previously mislabeled by machine learning models.


### Data Splitting
The dataset is split into three sets: training, validation, and test sets. The splitting ratio is 70% for training, 20% for validation, and 10% for testing. This ensures that the model is trained on a diverse set of images and evaluated on unseen data.

## Convolutional Neural Network (CNN) Architecture
The CNN architecture consists of convolutional layers followed by max-pooling layers and fully connected layers, designed to classify input images into two classes.

### Model Architecture
**Input Layer**: Accepts input images with three channels (RGB).

**Convolutional Layers**: Two convolutional layers with ReLU activation functions followed by batch normalization and max-pooling layers.

**Fully Connected Layers**: Two fully connected layers with ReLU activation functions, followed by a dropout layer to prevent overfitting.

**Output Layer**: A single neuron with a sigmoid activation function for binary classification.

### Training and Evaluation
* The code provides functions for training the CNN using stochastic gradient descent (SGD) and evaluating its performance on validation and test sets.
* Training includes monitoring loss and accuracy metrics, with early stopping implemented to prevent overfitting.
* Model hyperparameters such as batch size, learning rate, and momentum are adjustable.
* After training, the code evaluates the model's accuracy and generates plots for visualization, including training/validation accuracy and loss curves and a confusion matrix.

## Support Vector Machine (SVM) Model
For the SVM model, Histogram of Oriented Gradients (HOG) features are extracted from the grayscale satellite images.

### Model Training
After extracting HOG features from the grayscale satellite images, the SVM model is trained using these features as input. The SVM algorithm aims to find the optimal hyperplane that separates the feature space into different classes (ships and non-ships) while maximizing the margin between the classes.

### Hyperparameter Tuning
To optimize the performance of the SVM model, hyperparameters such as the regularization parameter (C) and kernel function type are tuned using grid search. 

## Model Comparison
The performance of the SVM model is compared with that of the Convolutional Neural Network (CNN) model trained on the same dataset using the various metrics and the confusion matrices.

### ConvNet Confusion Matrix
![ConvNet Confusion](https://github.com/JuFrei/ShipsFromOrbit/blob/main/img/ConvNet%20Conf.png)

### SVM Confusion Matrix
![SVM Confusion](https://github.com/JuFrei/ShipsFromOrbit/blob/main/img/SVM%20Conf.png)

### Convnet Misclassified Images
![ConvNet Misclassified](https://github.com/JuFrei/ShipsFromOrbit/blob/main/img/ConvNet%20Misclassified.png)

### SVM Misclassified Images
![SVM Misclassified](https://github.com/JuFrei/ShipsFromOrbit/blob/main/img/SVM%20Misclassified.png)


