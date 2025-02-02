# IMAGE-CLASSIFICATION-MODEL

**COMPANY** : CODTECH IT SOLUTIONS

**NAME** : GAURAV RAMASUBRAMANIAM

**INTERN ID** : CT08JUR

**DOMAIN** : MACHINE LEARNING

**DURATION** : 4 WEEKS

**MENTOR** : NEELA SANTOSH

# DESCRIPTION

# **CNN Image Classification using TensorFlow**  

## **📝 Project Overview**  
This project builds a **Convolutional Neural Network (CNN) model** for **image classification** using the **TensorFlow/Keras** framework. The model is trained on the **CIFAR-10 dataset**, which consists of **60,000 images** across **10 categories** (e.g., airplane, car, cat, dog, etc.). The trained model can predict the class of unseen images.  

---

## **📂 Dataset Information**  
We use the **CIFAR-10 dataset**, which contains:  
- **50,000 images** for training  
- **10,000 images** for testing  
- **10 classes**:  
  - ✈️ Airplane  
  - 🚗 Automobile  
  - 🐦 Bird  
  - 🐱 Cat  
  - 🦌 Deer  
  - 🐶 Dog  
  - 🐸 Frog  
  - 🐴 Horse  
  - 🚢 Ship  
  - 🚚 Truck  

The images are **32x32 pixels** with **3 color channels (RGB)**.

---

## **⚙️ Installation & Setup**  
### **1️⃣ Install Required Libraries**  
Ensure you have Python and the required dependencies installed:  
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### **2️⃣ Run the Jupyter Notebook**
Launch **Jupyter Notebook** and execute the provided code cells step by step.

```bash
jupyter notebook
```

---

## **🛠️ Model Architecture**  
The CNN model consists of the following layers:  
✔ **Conv2D (32 filters, 3x3, ReLU activation)**  
✔ **MaxPooling2D (2x2 pool size)**  
✔ **Conv2D (64 filters, 3x3, ReLU activation)**  
✔ **MaxPooling2D (2x2 pool size)**  
✔ **Flatten layer**  
✔ **Dense (128 neurons, ReLU activation)**  
✔ **Dropout (50%)**  
✔ **Dense (10 neurons, Softmax activation for classification)**  

---

## **🚀 Training & Evaluation**
The model is trained using:  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** **5 (adjustable)**  
- **Batch Size:** 32  

After training, the model is evaluated on the test dataset using:  
✔ **Accuracy Score**  
✔ **Confusion Matrix**  
✔ **Classification Report**  

---

## **📊 Results & Performance**
### **Training Performance**  
During training, we track:  
📌 **Accuracy** - Measures how well the model predicts correctly.  
📌 **Loss** - Measures the error during training.  
These metrics are plotted after training.

### **Model Evaluation**
After training, the model is tested on unseen data to check its accuracy. A **confusion matrix** is generated to visualize misclassifications.

---

## **📌 Predicting New Images**
To test the model on unseen images:  
1️⃣ Load a custom image.  
2️⃣ Resize it to **32x32 pixels**.  
3️⃣ Normalize pixel values to **0-1 range**.  
4️⃣ Predict using:  
```python
import cv2
image = cv2.imread('your_image.jpg')  # Load Image
image = cv2.resize(image, (32, 32))   # Resize
image = image / 255.0                 # Normalize
image = np.expand_dims(image, axis=0) # Add batch dimension
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print(f"Predicted Class: {class_names[predicted_class]}")
```

---

## **💾 Saving & Loading the Model**
Save the trained model:
```python
model.save("cnn_image_classifier.h5")
```
Load the model later for inference:
```python
from tensorflow.keras.models import load_model
model = load_model("cnn_image_classifier.h5")
```

---

## **📌 Future Improvements**
- **Data Augmentation** to improve generalization.  
- **More layers (deeper CNNs)** for better accuracy.  
- **Transfer Learning** with pre-trained models like **VGG16** or **ResNet**.  

---

## **📝 Conclusion**
This project demonstrates how to build and train a **CNN model for image classification** using **TensorFlow/Keras**. The model achieves good accuracy on the **CIFAR-10 dataset** and can be further optimized for real-world applications.  

---

# OUTPUT

