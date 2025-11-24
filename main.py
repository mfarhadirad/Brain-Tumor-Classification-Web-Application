from flask import Flask, render_template, request  
from PIL import Image  
import os  
import torch  
from torchvision import models  
from torchvision.models import  ResNet50_Weights

app = Flask(__name__)  

# Define the model paths and corresponding classes  
models_map = {  
    "AlexNet": ("alexnet", "D:/Githup/Tumor_detection/App/model_pth_files/AlexNet_95_97.pth"),  
    "GoogLeNet": ("googlenet", "D:/Githup/Tumor_detection/App/model_pth_files/GoogleNet_95_98.pth"),  
    "ResNet_50": ("resnet50", "D:/Githup/Tumor_detection/App/model_pth_files/ResNet_new_94_98.pth"),  
    "VGG_16": ("vgg16", "D:/Githup/Tumor_detection/App/model_pth_files/VGG_95_97.pth"),  
}  

loaded_models = {}  
for model_name, (model_fn_name, model_path) in models_map.items():  
    if model_fn_name == "alexnet":
 
        model = models.alexnet(weights=None)  # Use weights=None to avoid warnings  
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)  # Adjust to 4 classes  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        

# Assuming other parts of your code define `model_fn_name` and `model_path`  
    elif model_fn_name == "googleNet":

        model = models.vgg16(weights=None)  # Use weights=None to avoid warnings  
        model.fc = torch.nn.Linear(1024, 4)  # Adjust to 4 classes  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

    elif model_fn_name == "resnet50":


        model = models.resnet50(weights=None)  # Use weights=None to avoid warnings  
        model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust to 4 classes  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    elif model_fn_name == "vgg16":

        model = models.vgg16(weights=None)  # Use weights=None to avoid warnings  
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)  # Adjust to 4 classes  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    # Load the model state dict  
    model.eval()  
    loaded_models[model_name] = model  
  

@app.route('/', methods=['GET'])  
def html_rend():  
    return render_template('index.html', models=models_map.keys())  # Pass available models to template  

@app.route('/', methods=['POST'])  
def predict():  
    model_choice = request.form['model']  # Get the selected model from the form  
    imagefile = request.files['imagefile']  
    
    # Ensure the static/images directory exists  
    if not os.path.exists('static/images'):  
        os.makedirs('static/images')  

    # Save the image in the static/images folder  
    image_path = os.path.join("static", "images", imagefile.filename)  
    imagefile.save(image_path)  

    # Load and preprocess the image  
    weights = ResNet50_Weights.DEFAULT 
    preprocess = weights.transforms() 
    image = Image.open(image_path).convert('RGB')
    processed_image = preprocess(image)  
 

    # Choose the model for prediction  
    model = loaded_models[model_choice]  

    # Make prediction  
    with torch.inference_mode():
        unsqueezed_image = processed_image.unsqueeze(0)
        outputs = model(unsqueezed_image)  
    
    # Get the predicted class
    target_image_pred_probs = torch.softmax(outputs, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Load the class labels  
    with open("imagenet_classes.txt", encoding="utf-8") as f:  
        labels = [line.strip() for line in f.readlines()]  

    classification = f'{labels[target_image_pred_label]}'  

    return render_template('index.html', prediction=classification, image_file=imagefile.filename, models=models_map.keys(), selected_model=model_choice)  

if __name__ == '__main__':  
    app.run(port=3000, debug=True)