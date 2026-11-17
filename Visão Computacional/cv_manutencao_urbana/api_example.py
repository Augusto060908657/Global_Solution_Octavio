"""
Autor: Octavio Augusto Ramalho Brandão Pires | RM 555696
Exemplo de integração FastAPI (somente imagens):
  pip install fastapi uvicorn python-multipart
  uvicorn api_example:app --reload
  curl -F "file=@rua.jpg" "http://localhost:8000/predict"                # JSON com detecções
  curl -F "file=@rua.jpg" "http://localhost:8000/predict/image" -o out.png  # imagem anotada
"""
from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from cv_service import predict_urban_damage

app = FastAPI(title="CitizenTech - CV Manutenção Urbana (YOLO11n)")


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = None):
    try:
        r = predict_urban_damage(await file.read(), conf=conf)
        return r
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = None):
    try:
        r = predict_urban_damage(await file.read(), conf=conf, return_images=True)
        return Response(content=r["annotated_png"], media_type="image/png",
                        headers={"X-Num-Defects": str(r["num_defects"])})
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
