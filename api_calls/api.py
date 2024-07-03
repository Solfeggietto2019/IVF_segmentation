from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import json
from main import process, clean_up_files
import requests

is_busy = False

app = FastAPI()

# Define your API Key aquí (por simplicidad, hardcoded; usa variables de entorno en producción)
API_KEY = "f3c2e8d6b4a1d4e7c8a9b2d6e7f8c3a1b2d4e7c8a9b2d6f3c2e8d7b4a1c3d4"


class ProcessResult(BaseModel):
    json_output: str


# Función de verificación de la API Key
def get_api_key(api_key: Optional[str] = Header(None)):
    print(f"API Key recibida: {api_key}")  # Mensaje de depuración
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


@app.get("/is-busy")
async def return_busy():
    global is_busy
    return {"busy": is_busy}


@app.post("/process", response_model=ProcessResult)
async def process_video(
    json_file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    # Guardar el archivo JSON recibido
    global is_busy
    is_busy = True
    json_file_path = f"data/json/{json_file.filename}"
    with open(json_file_path, "wb") as f:
        shutil.copyfileobj(json_file.file, f)

    # Especificar una ruta para el archivo de video
    video_file_path = "data/video/test.mp4"

    # Llamar a la función de procesamiento
    json_output = process(json_file_path, video_file_path)

    # Realizar limpieza de archivos después de enviar la respuesta
    clean_up_files(json_file_path, video_file_path)

    url = "http://10.8.0.1/sofi-results/cloud-results/"
    data = json.loads(json_output)
    response = requests.post(url, json=data)

    is_busy = False

    return ProcessResult(json_output=json_output)


if __name__ == "__main__":
    # Ejecutar el servidor
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
