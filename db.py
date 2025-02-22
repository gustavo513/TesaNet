import psycopg2
import json

with open('config.json', 'r') as file:
    config = json.load(file)

db_config = config['postgres']

def conectar():
    return psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )

def consultar(query, params=None):
    try:
        with conectar() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                resultado = cursor.fetchall()
                return resultado if resultado else None  # Retorna None si no hay datos
    except psycopg2.Error as e:
        print(f"Error en la consulta: {e}")
        return None

def escribir(query, params=None):
    try:
        with conectar() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
    except psycopg2.Error as e:
        print(f"Error en la escritura: {e}")
        return False
    return True
