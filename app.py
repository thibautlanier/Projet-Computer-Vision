from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import numpy as np
from src.Models import NetworkColorWithScribble
from skimage import color
from torchvision.transforms import v2
import base64
import sys

app = Flask(__name__)

image_import = False 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NetworkColorWithScribble()
model.load_state_dict(torch.load('models/model1%.pth' if len(sys.argv) < 2 else sys.argv[1], map_location=device))
model.to(device)
model.eval()

# preprocessing (resize, convert in greyscale)
def preprocess_image(image, started=False):
    transform = v2.Compose([
        color.rgb2lab,
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Resize((128,128), antialias=True),
    ])
    image = image.convert('RGB')
    image = transform(image)
    if started:
        image[2, ...] = torch.zeros_like(image[2, ...])
        image[1, ...] = torch.zeros_like(image[1, ...])
    image = image.unsqueeze(0)

    return image.to(device)

@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        image_data = request.form['imageData']
        image_data = image_data.replace('data:image/png;base64,', '')

        # Sauvegarde l'image sur le serveur (ajuste le chemin selon tes besoins)
        with open('static/input_image.jpg', 'wb') as f:
           f.write(base64.b64decode(image_data))

        return 'Image saved on the server'
    except Exception as e:
        return 'Error saving image on the server', 500

def colorize_image(output_image):
    colorized_image = output_image[0].detach().moveaxis(0, 2).cpu()
    colorized_image_rgb = color.lab2rgb(colorized_image)
    return (colorized_image_rgb * 255).astype(np.uint8)

@app.route('/colorize', methods=['POST'])
def render_colorized_image():
    global image_import
    if not image_import:
        return render_template('index.html')
    image = Image.open('static/input_image.jpg')

    input_image = preprocess_image(image)
    with torch.no_grad():
        output = model(input_image)
    input_image = input_image.squeeze(0)
    colorized_image = colorize_image(output)
    colorized_image_path = 'static/colorized_image.jpg'
    colorized_image = Image.fromarray(colorized_image)
    colorized_image = colorized_image.resize((256,256))
    colorized_image.save(colorized_image_path)

    return jsonify({'colorized_image_path': colorized_image_path})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # retrieve uploaded image 

        file = request.files['file']
        if file:
            global image_import
            image_import = True
            image = Image.open(file)
            input_image = preprocess_image(image, True)
       
            input_image_path = 'static/input_image.jpg'

            input_image = input_image.squeeze().detach().moveaxis(0, 2).cpu().numpy()

            input_image = color.lab2rgb(input_image)

            input_image = Image.fromarray((input_image * 255).astype(np.uint8))
            
            input_image = input_image.resize((256,256))
            input_image.save(input_image_path)

            return render_template('index.html', input_image_path=input_image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
