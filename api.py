from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
    Depends,
    BackgroundTasks,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import json
import requests
from uuid import uuid4
from main import process, clean_up_files

is_busy = False
tasks = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your API Key aquí (por simplicidad, hardcoded; usa variables de entorno en producción)
API_KEY = "f3c2e8d6b4a1d4e7c8a9b2d6e7f8c3a1b2d4e7c8a9b2d6f3c2e8d7b4a1c3d4"


class MessageResponse(BaseModel):
    message: str
    task_id: str


class TaskResult(BaseModel):
    json_output: Optional[str] = None
    status: str


# Función de verificación de la API Key
def get_api_key(api_key: Optional[str] = Header(None)):
    print(f"API Key recibida: {api_key}")  # Mensaje de depuración
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


def process_and_cleanup(
    task_id: str,
    json_file_path: str,
    video_file_path: str,
    ids: List[int] = [],
    no_auto: Optional[bool] = False,
):
    global is_busy
    is_busy = True

    try:
        # Llamar a la función de procesamiento
        json_output = process(json_file_path, video_file_path, ids, no_auto)

        # Realizar limpieza de archivos después de enviar la respuesta
        clean_up_files(json_file_path, video_file_path)

        url = "http://10.8.0.1/sofi-results/cloud-results/"
        data = json.loads(json_output)
        response = requests.post(url, json=data)
        print(response)

        # Almacenar el resultado en el diccionario de tareas
        tasks[task_id] = {"status": "completed", "json_output": json_output}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "json_output": str(e)}
    finally:
        is_busy = False


@app.get("/is-busy")
async def return_busy():
    global is_busy
    return {"busy": is_busy}


@app.post("/process", response_model=MessageResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    ids_for_manual_processing: Optional[str] = Form(None),
    manual_processing: bool = Form(...),
    json_file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
):
    # Guardar el archivo JSON recibido
    json_file_path = f"data/json/{json_file.filename}"
    with open(json_file_path, "wb") as f:
        shutil.copyfileobj(json_file.file, f)

    # Especificar una ruta para el archivo de video
    video_file_path = "data/video/test.mp4"

    # Generar un task_id único
    task_id = str(uuid4())

    # Manejar el parámetro 'ids' y 'no_auto'
    # ids = ids_for_manual_processing
    if ids_for_manual_processing:
        ids = list(map(int, ids_for_manual_processing.split(",")))
    else:
        ids = []
    no_auto = manual_processing

    # Añadir la tarea de fondo para procesar y limpiar archivos
    background_tasks.add_task(
        process_and_cleanup, task_id, json_file_path, video_file_path, ids, no_auto
    )

    # Inicializar el estado de la tarea
    tasks[task_id] = {"status": "processing", "json_output": None}

    # Responder inmediatamente con un estado 200
    return {
        "message": "File received successfully, processing started.",
        "task_id": task_id,
    }


@app.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task_result(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResult(**task)


if __name__ == "__main__":
    # Ejecutar el servidor
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
