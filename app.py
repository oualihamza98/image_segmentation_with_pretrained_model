from flask import Flask, render_template, request, send_from_directory
import numpy as np
from PIL import Image
import cv2
import os
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

app = Flask(__name__, template_folder='templates')

# Path absolu pour le dossier images
IMAGES_FOLDER = "C:\\Users\\darkf\\OneDrive\\Desktop\\coding_work\\deployment_with_flask\\image_segmentation_app_with_DL\\images"

# Noms des 21 classes PASCAL VOC
CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'dining table', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train',
    'tv/monitor'
]

# Charger le modèle DeepLabV3 avec ResNet50 backbone
print("Chargement du modèle DeepLabV3 + ResNet50...")
model = deeplabv3_resnet50(pretrained=True)
model.eval()
print("Modèle chargé!")

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def segment():
    imagefile = request.files['imagefile']
    if not imagefile or not imagefile.filename:
        return render_template('index.html', prediction="No file selected")
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    
    # Sauvegarder l'image originale
    image_path = os.path.join(IMAGES_FOLDER, imagefile.filename)
    imagefile.save(image_path)
    
    # Charger et préparer l'image pour le modèle
    input_image = Image.open(image_path).convert('RGB')
    original_size = input_image.size
    
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Prédiction avec le modèle
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Créer le masque coloré (21 classes PASCAL VOC)
    colored_mask = create_pascal_label_colormap()[output_predictions]
    
    # Redimensionner à la taille originale
    colored_mask_pil = Image.fromarray(colored_mask).resize(original_size, Image.NEAREST)
    colored_mask = np.array(colored_mask_pil)
    
    # Charger l'image originale en array
    img_array = np.array(input_image)
    
    # Superposer avec l'image originale
    output = cv2.addWeighted(img_array, 0.6, colored_mask, 0.4, 0)
    
    # Sauvegarder l'image segmentée
    segmented_filename = "segmented_" + imagefile.filename
    segmented_path = os.path.join(IMAGES_FOLDER, segmented_filename)
    cv2.imwrite(segmented_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    
    # Récupérer les classes détectées
    unique_classes = np.unique(output_predictions)
    detected_classes = [CLASS_NAMES[cls] for cls in unique_classes if cls < len(CLASS_NAMES)]
    
    # Créer le message de prédiction
    num_classes = len(detected_classes)
    classes_str = ", ".join(detected_classes)
    prediction = f"DeepLabV3+ResNet50: {num_classes} classes detected"
    
    return render_template('index.html', 
                         prediction=prediction,
                         detected_classes=detected_classes,
                         segmented_image=segmented_filename)

def create_pascal_label_colormap():
    """Crée une palette de couleurs pour les 21 classes PASCAL VOC"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    # Classes PASCAL VOC
    colors = [
        [0, 0, 0],        # background
        [128, 0, 0],      # aeroplane
        [0, 128, 0],      # bicycle
        [128, 128, 0],    # bird
        [0, 0, 128],      # boat
        [128, 0, 128],    # bottle
        [0, 128, 128],    # bus
        [128, 128, 128],  # car
        [64, 0, 0],       # cat
        [192, 0, 0],      # chair
        [64, 128, 0],     # cow
        [192, 128, 0],    # dining table
        [64, 0, 128],     # dog
        [192, 0, 128],    # horse
        [64, 128, 128],   # motorbike
        [192, 128, 128],  # person
        [0, 64, 0],       # potted plant
        [128, 64, 0],     # sheep
        [0, 192, 0],      # sofa
        [128, 192, 0],    # train
        [0, 64, 128]      # tv/monitor
    ]
    
    for i, color in enumerate(colors):
        colormap[i] = color
    
    return colormap

# Route pour servir les images
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)