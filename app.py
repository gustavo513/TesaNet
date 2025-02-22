import os
import time
from flask import Flask, request, render_template, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import time
from werkzeug.security import generate_password_hash, check_password_hash
from db import consultar, escribir
from flask_login import LoginManager
from flask_login import login_user
from flask_login import UserMixin
from flask_login import login_required, logout_user, current_user

# Inicializar la app Flask
app = Flask(__name__)

#app.secret_key = os.urandom(24)
app.secret_key = 'tesanet2024'

# Cargar el modelo entrenado
model = load_model('tesanet-model.keras')
model.load_weights('tesanet-bif.weights.h5')
print(model.summary())

class User(UserMixin):
    def __init__(self, id, correo, nombre_usuario, nombre, apellido, fecha_creacion):
        self.id = id
        self.correo = correo
        self.nombre_usuario = nombre_usuario
        self.nombre = nombre
        self.apellido = apellido
        self.fecha_creacion = fecha_creacion
        

class Imagen():
    def __init__(self, nombre_imagen, tipo_imagen, fecha, tipo_neumonia, probabilidad):
        self.nombre_imagen = nombre_imagen
        self.tipo_imagen = tipo_imagen
        self.fecha = fecha
        self.tipo_neumonia = tipo_neumonia
        self.probabilidad = probabilidad


login_manager = LoginManager()
login_manager.login_view = 'iniciarSesion'
login_manager.init_app(app)

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

    start_time = time.time()
    prediction = model.predict(image)
    end_time = time.time()

    # Tiempo de inferencia en segundos
    inference_time = end_time - start_time
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

    return prediction


@login_manager.user_loader
def load_user(user_id):
    usuario = consultar('SELECT id_usuario, correo, nombre_usuario, nombre, apellido, fecha_creacion FROM public."Usuario" WHERE id_usuario = %s;', (user_id,))
    if usuario:
        return User(
                    id=usuario[0][0], 
                    correo=usuario[0][1], 
                    nombre_usuario=usuario[0][2],
                    nombre=usuario[0][3],
                    apellido=usuario[0][4],
                    fecha_creacion=usuario[0]
                )
    return None


@app.route('/')
def iniciar_sesion():
    return render_template('iniciarSesion.html')


@app.route('/', methods=['POST'])
def inic_ses_post():

    correo = request.form.get("correo")
    contraseña = request.form.get("contraseña")

    usuario = consultar(
        'SELECT id_usuario, correo, nombre_usuario, contraseña FROM public."Usuario" WHERE correo = %s OR nombre_usuario = %s;',
        (correo, correo)
    )

    if not usuario or not check_password_hash(usuario[0][3], contraseña):
        flash("Fallo al iniciar sesion. Verifique sus credenciales")
        return redirect(url_for('iniciar_sesion'))
    
    usuario_obj = User(
                    id=usuario[0][0], 
                    correo=usuario[0][1], 
                    nombre_usuario=usuario[0][2],
                    nombre=None,
                    apellido=None,
                    fecha_creacion=None
                )
    login_user(usuario_obj, remember=False)
    #return render_template('index.html')
    return redirect(url_for('cargar_imagen')) 


@app.route('/cargar_imagen')
@login_required
def cargar_imagen():
    return render_template('cargarImagen.html')


@app.route('/registro')
def registro():
    return render_template('registro.html')


@app.route('/registro', methods=['POST'])
def registro_post():
    nombre = request.form.get("nombre")
    apellido = request.form.get("apellido")
    correo = request.form.get("correo")
    nombre_usuario = request.form.get("nombre_usuario")
    password = request.form.get("contraseña")

    usuario = consultar(
        'SELECT nombre, apellido, correo, nombre_usuario, contraseña FROM public."Usuario" WHERE correo = %s;',
        (correo,)
    )

    if usuario:
        return redirect(url_for('iniciar_sesion')) 

    hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

    nuevo_usuario = escribir(
        'INSERT INTO public."Usuario" (nombre, apellido, correo, nombre_usuario, contraseña) VALUES (%s, %s, %s, %s, %s);',
        (nombre, apellido, correo, nombre_usuario, hashed_password)
    )

    return redirect(url_for('iniciar_sesion'))    


@app.route('/perfil')
@login_required
def perfil():
    
    usuario = consultar(
        'SELECT id_usuario, correo, nombre_usuario, nombre, apellido, fecha_creacion FROM public."Usuario" WHERE id_usuario = %s;', 
        (current_user.id,)
    )

    usuario_obj = User(
                    id=usuario[0][0], 
                    correo=usuario[0][1], 
                    nombre_usuario=usuario[0][2],
                    nombre=usuario[0][3],
                    apellido=usuario[0][4],
                    fecha_creacion=usuario[0][5]
    )

    return render_template('perfil.html', usuario=usuario_obj)


@app.route('/perfil', methods=['POST'])
@login_required
def guardar_perfil():

    id = request.form.get('id')
    nombre = request.form.get('nombre')
    apellido = request.form.get('apellido')
    correo = request.form.get('correo')
    nombre_usuario = request.form.get('nombre_usuario')

    escribir(
        'UPDATE public."Usuario" SET nombre = %s, apellido = %s, correo = %s, nombre_usuario = %s WHERE id_usuario = %s',
        (nombre, apellido, correo, nombre_usuario, id)
    )

    return redirect(url_for('cargar_imagen'))


@app.route('/eliminar')
@login_required
def eliminar_perfil():

    escribir(
        'UPDATE public."Usuario" SET correo = NULL, nombre_usuario = NULL, contraseña = NULL, nombre = NULL, apellido = NULL, estado = 0 WHERE id_usuario = %s;',
        (current_user.id,)
    )

    logout_user()
    return redirect(url_for('iniciar_sesion'))


@app.route('/historial')
@login_required
def historial():
    imagenes = consultar(
        'SELECT nombre_imagen, tipo_imagen, fecha_carga, tipo_neumonia, probabilidad FROM public."Imagen" WHERE id_usuario = %s ORDER BY id_imagen DESC;',
        (current_user.id,)
    )

    return render_template('historial.html', imagenes=imagenes)


@app.route('/cerrar_sesion')
@login_required
def cerrarSesion():
    logout_user()
    return redirect(url_for('iniciar_sesion'))


# Ruta para cargar la imagen
@app.route('/resultado', methods=['POST'])
@login_required
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    id_imagen = consultar(
        'SELECT COALESCE(MAX(id_imagen), 0) + 1 FROM public."Imagen";',
        None
    )

    nombre_imagen = "imagen"+str(id_imagen[0][0])

    tipo_imagen = file.filename.split(".")[1]

    nombre_dir = nombre_imagen+"."+tipo_imagen
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], nombre_dir)
    file.save(file_path)  # Guardar la imagen en la carpeta 'uploads'
    print(f"Imagen guardada en: {file_path}")
    
    # Realizar predicción
    prediction = predict_image(file_path)
    print(f"Predicciones (probabilidades): {prediction}")
    
    # Interpretar la predicción
    class_labels = ['Normal', 'Neumonía Viral', 'Neumonía Bacteriana']  # Ajusta según tus clases
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100  # Obtener la confianza para la clase predicha
    
    # Crear una etiqueta con la clase predicha y el porcentaje
    prediction_label = f"{class_labels[predicted_class]} ({confidence:.2f}%)"
    print(f'Resultado: {prediction_label}')
    
    # Asegurar que se cargue la imagen correcta añadiendo una marca de tiempo
    image_url = url_for('static', filename='uploads/' + nombre_dir) + "?t=" + str(int(time.time()))

    tipo_neumonia = {"Normal": "N", "Neumonía Viral": "V", "Neumonía Bacteriana": "B"}

    escribir(
        'INSERT INTO public."Imagen"(nombre_imagen, tipo_imagen, tipo_neumonia, probabilidad, id_usuario) VALUES (%s, %s, %s, %s, %s)',
        (nombre_imagen, tipo_imagen, tipo_neumonia[class_labels[predicted_class]], confidence, current_user.id)
    )
    
    # Renderizar el template con la predicción
    return render_template('cargarImagen.html', prediction=prediction_label, image=image_url)

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
    app.run(host='0.0.0.0', port=5000)

