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
```
usedClass = [
 'Apple__black_rot',
 'Apple__healthy',
 'Apple__rust',
 'Apple__scab',
#  'Cassava__bacterial_blight',
#  'Cassava__brown_streak_disease',
#  'Cassava__green_mottle',
#  'Cassava__healthy',
#  'Cassava__mosaic_disease',
#  'Cherry__healthy',
#  'Cherry__powdery_mildew',
 'Chili__healthy',
 'Chili__leaf curl',
 'Chili__leaf spot',
 'Chili__whitefly',
 'Chili__yellowish',
#  'Coffee__cercospora_leaf_spot',
#  'Coffee__healthy',
#  'Coffee__red_spider_mite',
#  'Coffee__rust',
 'Corn__common_rust',
 'Corn__gray_leaf_spot',
 'Corn__healthy',
 'Corn__northern_leaf_blight',
 'Cucumber__diseased',
 'Cucumber__healthy',
 'Gauva__diseased',
 'Gauva__healthy',
 'Grape__black_measles',
 'Grape__black_rot',
 'Grape__healthy',
 'Grape__leaf_blight_(isariopsis_leaf_spot)',
#  'Jamun__diseased',
#  'Jamun__healthy',
 'Lemon__diseased',
 'Lemon__healthy',
 'Mango__diseased',
 'Mango__healthy',
#  'Peach__bacterial_spot',
#  'Peach__healthy',
 'Pepper_bell__bacterial_spot',
 'Pepper_bell__healthy',
#  'Pomegranate__diseased',
#  'Pomegranate__healthy',
 'Potato__early_blight',
 'Potato__healthy',
 'Potato__late_blight',
 'Rice__brown_spot',
 'Rice__healthy',
 'Rice__hispa',
 'Rice__leaf_blast',
 'Rice__neck_blast',
 'Soybean__bacterial_blight',
 'Soybean__caterpillar',
 'Soybean__diabrotica_speciosa',
 'Soybean__downy_mildew',
 'Soybean__healthy',
 'Soybean__mosaic_virus',
 'Soybean__powdery_mildew',
 'Soybean__rust',
 'Soybean__southern_blight',
#  'Strawberry___leaf_scorch',
#  'Strawberry__healthy',
 'Sugarcane__bacterial_blight',
 'Sugarcane__healthy',
 'Sugarcane__red_rot',
 'Sugarcane__red_stripe',
 'Sugarcane__rust',
#  'Tea__algal_leaf',
#  'Tea__anthracnose',
#  'Tea__bird_eye_spot',
#  'Tea__brown_blight',
#  'Tea__healthy',
#  'Tea__red_leaf_spot',
 'Tomato__bacterial_spot',
 'Tomato__early_blight',
 'Tomato__healthy',
 'Tomato__late_blight',
 'Tomato__leaf_mold',
 'Tomato__mosaic_virus',
 'Tomato__septoria_leaf_spot',
 'Tomato__spider_mites_(two_spotted_spider_mite)',
 'Tomato__target_spot',
 'Tomato__yellow_leaf_curl_virus',
 'Wheat__brown_rust',
 'Wheat__healthy',
 'Wheat__septoria',
 'Wheat__yellow_rust']
```

## **🔧 Installation**

Clone the repository:  
```git clone https://github.com/ATHARV_CHAUDHAR/plant-disease-detection.git```

1. ```cd plant-disease-detection```  
2. Install required dependencies:  
   ```pip install \-r requirements.txt```  
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

