# Detection of Lung Infection
<src = "img p-5.jpg" alt="Mercedes-Benz">

## Description

Artificial Intelligence (AI) has made significant advancements and has demonstrated its capability to tackle complex problems that traditionally required human expertise. One prominent domain where AI has shown great potential is healthcare.

In healthcare, extensive research is conducted daily to harness the power of deep learning algorithms for the betterment of humanity. Deep learning techniques have shown promising results in various healthcare applications, including disease diagnosis and medical image analysis.

This project focuses on the detection of lung infection using a convolutional neural network (CNN). By leveraging AI and medical imagery, the goal is to develop a model that can accurately classify lung infection in individuals. The dataset used for training and testing contains images categorized into three classes: healthy, type 1 disease, and type 2 disease.

The project encompasses various tasks, including data preprocessing, data augmentation, building and training CNN models, and evaluating their performance based on metrics such as accuracy, precision, recall, and F1-score. Additionally, transfer learning techniques using pre-trained models such as MobileNet and DenseNet121 are employed to enhance the model's performance.

The ultimate aim of this project is to contribute to the advancement of AI in healthcare by developing an effective model for the early detection of lung infection, thereby facilitating timely medical interventions and improving patient outcomes.


## Objective

To build a model using a convolutional neural network that can classify lung infection in a person using medical imagery.

## Dataset Description

The dataset contains three different classes, including healthy, type 1 disease, and type 2 disease.

- Train folder: This folder has images for training the model, which is divided into subfolders having the same name as the class. 
- Test folder: This folder has images for testing the model, which is divided into subfolders having the same name as the class.

## Operations to be Performed

Following operations should be performed using Keras or PyTorch or TorchVision:

1. Import the necessary libraries.
2. Plot the sample images for all the classes.
3. Plot the distribution of images across the classes.
4. Build a data augmentation for the train data to create new data with translation, rescale, flip, and rotation transformations. Rescale the image at 48x48.
5. Build a data augmentation for the test data to create new data and rescale the image at 48x48.
6. Read images directly from the train folder and test folder using the appropriate function.

## Build 3 CNN Models

For each CNN model:

1. Add convolutional layers with different filters, max pool layers, dropout layers, and batch normalization layers.
2. Use Relu as an activation function.
3. Take the loss function as categorical cross-entropy.
4. Take rmsprop as an optimizer.
5. Use early stopping with the patience of two epochs and monitor the validation loss or accuracy.
6. Train the model using a generator and test the accuracy of the test data at every epoch.
7. Plot the training and validation accuracy and the loss.
8. Observe the precision, recall, and F1-score for all classes for both grayscale and color models and determine if the model's classes are good.

## Transfer Learning using MobileNet

1. Prepare data for the pre-trained MobileNet model, with color mode as RGB.
2. Create an instance of a MobileNet pre-trained model.
3. Add a dense layer, dropout layer, and batch normalization layer on the pre-trained model.
4. Create a final output layer with a SoftMax activation function.
5. Change the batch size, activation function, and optimizer to rmsprop and observe if the accuracy increases.
6. Take the loss function as categorical cross-entropy.
7. Use early stopping with the patience of two epochs and a callback function for preventing overfitting.
8. Train the model using a generator and test the accuracy of the test data at every epoch.
9. Plot the training and validation accuracy and the loss.
10. Observe the precision, recall, and F1-score for all classes for both grayscale and color models and determine if the model's classes are good.

## Transfer Learning using Densenet121

1. Prepare the dataset for the transfer learning algorithm using Densenet121 with the image size as 224x224x3.
2. Freeze the top layers of the pre-trained model.
3. Add a dense layer at the end of the pre-trained model followed by a dropout layer and try various combinations to get accuracy.
4. Add the final output layer with a SoftMax activation function.
5. Take the loss function as categorical cross-entropy.
6. Take Adam as an optimizer.
7. Use early stopping to prevent overfitting.
8. Train the model using the generator and test the accuracy of the test data at every epoch.
9. Plot the training and validation accuracy and the loss.
10. Observe the precision, recall, and F1-score for all classes for both grayscale and color models and determine if the model's classes are good.

## Final Step

Compare all the models based on accuracy, precision, recall, and F1-score.
