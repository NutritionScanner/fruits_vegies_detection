# 🍏 Fruits and Veggies Detection System

## 📜 Overview

The **Fruits and Veggies Detection System** is a Convolutional Neural Network (CNN) based model designed to detect and classify various types of fruits from images. Trained on a comprehensive dataset of fruit images, this application is capable of providing **real-time predictions** using a Image or Camera. It is designed for diverse scenarios, including identifying fruits in markets, automating sorting systems, and serving as a tool for nutritional analysis.

## 📊 Model Prediction Performance

Our 36-class Fruits and Vegetables Detection Model has been evaluated using a Confusion Matrix Heatmap! 🌟 Here's how the model performed:

![🟦 Confusion Matrix Heatmap](./Prediction_heatmap.png)

### 🔍 Key Highlights:
- ✅ **Accurate Predictions**: Most classes are correctly identified, as seen along the diagonal.  
- ⚠️ **Misclassifications**: Some overlap exists for similar items (e.g., [mention example if needed]).  
- 📈 **Performance Insights**: The heatmap provides a clear overview of the model's strengths and areas for improvement.  

This visual representation helps us understand the model's behavior and refine it further! 🚀  

## 🌟 Features

- 🔍 **Real-Time Detection**: Utilizes a webcam to detect and classify fruits dynamically.
- 🎯 **Good Accuracy**: Achieves impressive accuracy with minimal overfitting through techniques such as data augmentation, regularization, and adaptive learning rate adjustments.

## ⚙️ Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- Numpy

To install the required packages, use the package manager [pip](https://pip.pypa.io/en/stable/):

```bash
pip install tensorflow opencv-python numpy
```

## 🏋️‍♂️ Model Training
The model was trained on a dataset of fruit images using a Convolutional Neural Network (CNN). Training was performed over multiple epochs with various strategies to enhance accuracy and generalization:

- **Data Augmentation:** Applied rotations, shifts, and flips to the training images to increase model robustness.
- **Regularization:** Utilized L2 regularization and dropout techniques to prevent overfitting.
- **Learning Rate Scheduler:** Dynamically adjusted the learning rate during training for smooth convergence.

## 📈 Training Results
- Below are the results of the last 5 epochs:
```
Epoch 15/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 151s 2s/step - accuracy: 0.9757 - loss: 0.1857 - val_accuracy: 0.9658 - val_loss: 0.1586
Epoch 16/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 132s 2s/step - accuracy: 0.9800 - loss: 0.1771 - val_accuracy: 0.9715 - val_loss: 0.1550
Epoch 17/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 141s 2s/step - accuracy: 0.9781 - loss: 0.1624 - val_accuracy: 0.9715 - val_loss: 0.1506
Epoch 18/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 143s 2s/step - accuracy: 0.9820 - loss: 0.1529 - val_accuracy: 0.9687 - val_loss: 0.1433
Epoch 19/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 141s 2s/step - accuracy: 0.9849 - loss: 0.1405 - val_accuracy: 0.9687 - val_loss: 0.1470
Epoch 20/20
49/49 ━━━━━━━━━━━━━━━━━━━━ 114s 2s/step - accuracy: 0.9869 - loss: 0.1287 - val_accuracy: 0.9744 - val_loss: 0.1352
```
