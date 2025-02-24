# TesaNet

Identificador de neumonías virales y bacterianas.

## Descripción

TesaNet es una aplicación que utiliza un modelo de aprendizaje automático para identificar neumonías virales y bacterianas en imágenes radiológicas. Este proyecto permite a los usuarios cargar imágenes y obtener resultados de clasificación.

## Requisitos

Asegúrate de tener instalado [Python](https://www.python.org/downloads/) en tu máquina. Se recomienda utilizar un entorno virtual para gestionar las dependencias.

## Instalación

1. **Clonar el repositorio**:
   
   ```bash
   git clone https://github.com/gustavo513/TesaNet.git
   cd TesaNet

3. **Crear un entorno virtual**

   python -m venv .venv
   
   source .venv/bin/activate  # En Linux/Mac
   
   .venv\Scripts\activate     # En Windows

   [El entorno virtual debe activarse siempre antes de ejecutar el proyecto o después de reabrir el editor de código, ya sea en Linux o Windows]

4. **Instalar las dependencias**
   
    pip install -r requirements.txt

5. **Crear la carpeta /TesaNet/static/uploads para guardar las imágenes cargadas**

    cd static

    mkdir uploads

6. **Importar el archivo tesanet_bd.sql desde PostgreSQL para crear la base de datos**

   Directorio: ../tesanet/tesanet_bd.sql
   
7. **Ejecutar la aplicación**
   
    python app.py
