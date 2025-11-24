
<img width="1918" height="1087" alt="Screenshot 2025-11-24 093832" src="https://github.com/user-attachments/assets/b60adf57-17a7-478f-a3e9-bfde77e8188b" />

# Brain-Tumor-Classification-Web-Application
A deep learning-powered web application built with Flask that performs multi-class brain tumor classification from MRI scans. This tool demonstrates the practical deployment of multiple convolutional neural network (CNN) architectures for medical image analysis, achieving high diagnostic accuracy across different model implementations.

## 🎯 Key Features
Multi-Model Architecture: Implements and compares four state-of-the-art CNN architectures:

AlexNet (97.33% Accuracy, 97.61% Sensitivity, 98.28% Precision, 98.31% Specificity, 97.94% F1-Score)

VGG-16 (97.33% Accuracy, 99.31% Sensitivity, 97.00% Precision, 96.94% Specificity, 98.14% F1-Score)

GoogLeNet (98.09% Accuracy, 99.67% Sensitivity, 97.10% Precision, 96.89% Specificity, 98.37% F1-Score)

ResNet-50 (98.03% Accuracy, 100% Sensitivity, 97.40% Precision, 97.31% Specificity, 98.68% F1-Score)

## Note: Models are available through this link:


## Note: Augmented data are available through this link:
https://drive.google.com/drive/folders/152IJwaDzB-6Gs8JsWt1X2KqgGjYf1f4j?usp=sharing

## User-Friendly Web Interface: Intuitive Flask web interface allowing users to:

Upload MRI images in JPEG format

Select the model from multiple pre-trained models

View instant classification results with the uploaded image display

Production-Ready Deployment: Configured for local deployment with Flask server, ready for containerization or cloud deployment.

## 🛠️ Technical Implementation
Backend Framework: Flask (Python)
Deep Learning: PyTorch with torchvision
Computer Vision: PIL/Pillow for image preprocessing
Model Architectures: Custom fine-tuned versions of AlexNet, VGG-16, GoogLeNet, ResNet-50
Frontend: HTML/CSS with Jinja2 templating

## 🚀 Installation & Usage
Clone the repository

bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
python app.py
Access the web interface

Open your browser to http://localhost:3000

Upload an MRI image and select a model for classification

## 🔬 Clinical Relevance
This application demonstrates the practical implementation of AI-assisted diagnostic tools in radiology. The high accuracy metrics across multiple architectures highlight the potential of deep learning to support clinical decision-making in neuroimaging and oncology.

## 📈 Future Enhancements
Integration of 3D CNN models for volumetric analysis

Implementation of uncertainty quantification for predictions

Development of DICOM support for direct PACS integration

Addition of explainable AI (XAI) features for model interpretability

This project showcases the intersection of clinical radiology knowledge and advanced deep learning implementation, creating a bridge between AI research and practical medical applications.

Connect with me:  LinkedIn: [https://www.linkedin.com/mohammad-farhadi-rad-927ab7284/](https://www.linkedin.com/in/mohammad-farhadi-rad-927ab7284/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BN6x7gN78Sd2E%2FeO7Q0nIqw%3D%3D)






