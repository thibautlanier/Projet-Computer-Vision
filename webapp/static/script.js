
function colorizeImage() {  
    $.ajax({url: '/colorize', type: 'POST', 
    success: function (response) {
        let colorizedImage = document.getElementById('colorizedImage');
        colorizedImage.src = response.colorized_image_path + '?random=' + new Date().getTime();;
        colorizedImage.style.display = 'inline';
    }});
}

let currentColor = '#000000'; 
let modalCanvas = document.getElementById('modalCanvas');
let context = modalCanvas.getContext('2d');
let painting = false;

function openModal() {
    let modal = document.getElementById('myModal');
    modal.style.display = 'block';
    

    var inputImage = document.getElementById('inputImage');
    var imageWidth = inputImage.width;
    var imageHeight = inputImage.height;


    modalCanvas.width = imageWidth;
    modalCanvas.height = imageHeight;
    
    context.clearRect(0, 0, modalCanvas.width, modalCanvas.height);
    
    context.drawImage(inputImage, 0, 0, imageWidth, imageHeight);
}


function saveImage() {
    var inputImage = document.getElementById('inputImage');
    var modalCanvas = document.getElementById('modalCanvas');
    imageDataUrl = modalCanvas.toDataURL('image/png');
    inputImage.src = imageDataUrl;
    closeModal();
    $.ajax({
        type: 'POST',
        url: '/save_image',  
        data: {
            imageData: imageDataUrl
        },
        success: function(response) {
            console.log(response);
        },
        error: function(error) {
            console.error(error);

        }
    });
}

function closeModal() {
 
    document.getElementById('myModal').style.display = 'none';
}

function setColor(event) {
    let colorPicker = document.getElementById('colorPicker');
    let selectedColor = colorPicker.value;
    currentColor = selectedColor;
}


function startPainting() {
    painting = true;
}

function stopPainting() {
    painting = false;
}

function paintImage(event) {
    if (!painting) return;

    var x = event.clientX - modalCanvas.getBoundingClientRect().left;
    var y = event.clientY - modalCanvas.getBoundingClientRect().top;
    context.fillStyle = currentColor;
    context.fillRect(x, y, 1, 1); 
}