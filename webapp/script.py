from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from network import NetworkColor
from skimage import color
from matplotlib import pyplot as plt
from torchvision.transforms import v2
app = Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NetworkColor()
model.load_state_dict(torch.load('../model.pth'))
model.to(device)
model.eval()

# preprocessing (resize, convert in greyscale)
def preprocess_image(image):
    transform = v2.Compose([
        color.rgb2lab,
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Resize((128,128), antialias=True),
    ])
    
    image = transform(image)
    image = image[0,...].unsqueeze(0).unsqueeze(0)
    print(image.shape)
    return image.to(device)



def colorize_image(input_image, output_image):
    colorized_image = torch.cat((input_image, output_image[0]), dim=0)
    colorized_image = colorized_image.detach().moveaxis(0, 2).cpu()
    colorized_image_rgb = color.lab2rgb(colorized_image)
    return (colorized_image_rgb * 255).astype(np.uint8)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # retrieve uploaded image 
        file = request.files['file']
        if file:
            
            image = Image.open(file)
            input_image = preprocess_image(image)
       
            with torch.no_grad():
                output = model(input_image)
            input_image = input_image.squeeze(0)
      

            colorized_image_path = 'static/colorized_image.jpg'
            colorized_image = colorize_image(input_image, output)
            colorized_image = Image.fromarray(colorized_image)
            colorized_image = colorized_image.resize((256,256))
            colorized_image.save(colorized_image_path)

            input_image_path = 'static/input_image.jpg'
            input_image = input_image.squeeze().detach().cpu().numpy()
            input_image = input_image.astype(np.uint8)
            input_image = Image.fromarray(input_image)
            input_image = input_image.resize((256,256))
            input_image.save(input_image_path)

            return render_template('index.html', colorized_image_path=colorized_image_path, input_image_path=input_image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
