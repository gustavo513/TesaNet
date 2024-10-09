import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2 as cv
import time

# Inicializar la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('tesanet-model.keras')
model.load_weights('tesanet-bif.weights.h5')
print(model.summary())

# Ruta a la carpeta de uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Extensiones de archivo permitidas
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Definir una función de predicción
def predict_image(file_path):
    # Cargar la imagen usando OpenCV
    image = cv.imread(file_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        raise ValueError(f"Error al cargar la imagen: {file_path}. Asegúrate de que la ruta sea correcta y que el archivo exista.")

    # Redimensionar la imagen
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_LINEAR)

    # Convertir la imagen a escala de grises
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = image / 255.0
    
    image = np.array(image)

    # Expandir dimensiones para batch
    image = np.expand_dims(image, axis=-1)  # Añade la dimensión de canal
    image = np.expand_dims(image, axis=0)   # Añade la dimensión de batch

    prediction = model.predict(image)

    return prediction

# Ruta de la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cargar la imagen
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)  # Guardar la imagen en la carpeta 'uploads'
    print(f"Imagen guardada en: {file_path}")
    
    # Realizar predicción
    prediction = predict_image(file_path)
    print(prediction)
    
    # Interpretar la predicción
    class_labels = ['Normal', 'Neumonía Viral', 'Neumonía Bacteriana']  # Ajusta según tus clases
    predicted_class = np.argmax(prediction, axis=1)
    print('Resultado: ', predicted_class)
    prediction_label = class_labels[predicted_class[0]]
    print(prediction_label)

    # Asegurar que se cargue la imagen correcta añadiendo una marca de tiempo
    image_url = url_for('static', filename='uploads/' + file.filename) + "?t=" + str(int(time.time()))
    
    # Renderizar el template con la predicción
    return render_template('index.html', prediction=prediction_label, image=image_url)

# Deshabilitar el caché en todas las respuestas
@app.after_request
def add_header(response):
    # Deshabilitar el caché
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

print(tf.__version__)
print(tf.keras.__version__)
print(cv.__version__)

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)

