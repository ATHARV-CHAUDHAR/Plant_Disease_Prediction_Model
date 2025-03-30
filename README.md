# **Plant Disease Detection using Machine Learning**

## **📌 Project Overview**

This project utilizes deep learning techniques to detect plant diseases from images. It employs Convolutional Neural Networks (CNNs) and Transformer-based Vision models (ViT) to classify plant leaf images into healthy and diseased categories.

## **👨‍💻 Author**

**Atharv Chaudhari**

## **🚀 Features**

* **Deep Learning Models**: Uses CNN and ViT (Vision Transformer) for plant disease classification.  
* **Data Augmentation**: Implements `ImageDataGenerator` to enhance model generalization.  
* **Visualization**: Provides confusion matrix and classification reports for evaluation.  
* **Early Stopping & Model Checkpoints**: Prevents overfitting and saves the best model.

## **🛠️ Tech Stack**

* **Python**  
* **TensorFlow & Keras**  
* **OpenCV & Matplotlib**  
* **Seaborn & Scikit-learn**  
* **Vision Transformer (ViT B16)**

## **📂 Dataset**

The dataset contains images of various plant leaves categorized into healthy and diseased classes. The dataset is preprocessed and split into training and validation sets.

## **🔧 Installation**

Clone the repository:  
```git clone https://github.com/yourusername/plant-disease-detection.git```

1. cd plant-disease-detection  
2. Install required dependencies:  
   pip install \-r requirements.txt  
3. (Optional) If using Jupyter Notebook, start the notebook:  
   jupyter notebook

## **📊 Model Training & Evaluation**

1. **Data Preprocessing**  
   * Image resizing, augmentation, and train-validation split.  
2. **Model Training**  
   * Trains a CNN/Vision Transformer model with pre-trained weights.  
3. **Model Evaluation**  
   * Generates confusion matrix and classification reports.

## **📸 Sample Visualizations**

* Displays sample images with augmentation.  
* Plots loss & accuracy graphs for model training.

## **📈 Results & Performance**

* The trained model achieves high accuracy in detecting plant diseases.  
* Uses transfer learning to improve efficiency.

## **🤖 How to Run**

1. Run the notebook `plant-disease-detection.ipynb` step by step.  
2. Ensure dataset images are correctly placed in the specified directory.

## **🔗 References**

* [TensorFlow Documentation](https://www.tensorflow.org/)  
* ViT Model

---

Feel free to contribute, raise issues, or fork this repository\!

