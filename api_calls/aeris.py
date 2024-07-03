import requests
import base64
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import io

# Ajuste del api_key y la ruta de la imagen
api_key = "InS_0EIY4dY8vssvD-0sjwRL5nBnfcTi"


def convert_image_to_base64(image_path):
    # Abrir la imagen usando Pillow
    with Image.open(image_path) as img:
        # Asegurarse de que la imagen esté en formato RGB
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print(f"Tamaño de la imagen codificada: {len(b64_string)} caracteres")
        return b64_string


def api_maturity(image_path, api_key):
    # URL del servidor
    url = "https://aeris.ivf20.app/evaluate"

    # Convertir la imagen a base64
    img_b64 = convert_image_to_base64(image_path)

    # Crear el objeto JSON
    payload = {"image": img_b64, "num_groups": 1}

    json_payload = json.dumps(payload)
    print(
        f"Payload enviado: {json_payload[:500]}..."
    )  # Muestra los primeros 500 caracteres del payload

    headers = {"Content-Type": "application/json", "api-key": api_key}

    # Realiza la solicitud POST
    response = requests.post(url, data=json_payload, headers=headers)

    print(f"Respuesta del servidor: {response.status_code}")
    print(f"Contenido de la respuesta: {response.text}")

    if response.status_code == 200:
        print("Respuesta recibida correctamente.")
        return response.json()
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(response.text)
        return None


def save_response_to_json(response, filename="./results.json"):
    with open(filename, "w") as json_file:
        json.dump(response, json_file, indent=4)
        print(f"Resultados guardados en {filename}")


def get_egg_features(response):
    initial_egg_values = {
        "cyto_minor_minoraxis": 0,
        "cyto_majoraxis": 0,
        "cyto_eccentricity": 0,
        "cyto_area": 0,
        "granu_minor_minoraxis": 0,
        "granu_majoraxis": 0,
        "granu_eccentricity": 0,
        "granu_area": 0,
        "granu_area_rel": 0,
        "polarbody_minor_minoraxis": 0,
        "polarbody_majoraxis": 0,
        "polarbody_eccentricity": 0,
        "polarbody_area": 0,
        "peri_cyto_minoraxis": 0,
        "peri_cyto_majoraxis": 0,
        "peri_cyto_eccentricity": 0,
        "peri_cyto_area": 0,
        "zp_peri_cyto_minoraxis": 0,
        "zp_peri_cyto_majoraxis": 0,
        "zp_peri_cyto_eccentricity": 0,
        "zp_peri_cyto_area": 0,
    }

    # Extraer las características del primer oocyte
    oocyte_features = response["oocytes"][0]["features"]

    # Crear un nuevo diccionario combinando las claves
    final_egg_features = initial_egg_values.copy()
    for key, value in oocyte_features.items():
        if key in final_egg_features:
            final_egg_features[key] = value

    return final_egg_features

def main():
    api_maturity("./inyected_egg_1719960912.png",api_key)

if __name__ == "__main__":
    main()