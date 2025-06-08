# from fastapi import FastAPI, File, UploadFile, Form, Request
# from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
# from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input  # type: ignore
# from tensorflow.keras.models import Model  # type: ignore
# import joblib
# import numpy as np
# import os
# import base64
# import uuid
# import json

# app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # For dev; in production use ["http://localhost:8000"] or specific domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load ML Models
# rf = joblib.load(os.path.join(BASE_DIR, "random_forest_dr_model.pkl"))
# le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
# base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
# feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# # Feedback folders
# feedback_image_folder = os.path.join(BASE_DIR, "feedbacks/images")
# feedback_data_folder = os.path.join(BASE_DIR, "feedbacks/data")
# os.makedirs(feedback_image_folder, exist_ok=True)
# os.makedirs(feedback_data_folder, exist_ok=True)

# # Serve static and template files
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
# templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# # Homepage route
# @app.get("/", response_class=HTMLResponse)
# async def serve_frontend(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Prediction route
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     img_path = "temp_image.jpg"
#     with open(img_path, "wb") as f:
#         f.write(await file.read())

#     img = load_img(img_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)

#     feature = feature_extractor.predict(img_array, verbose=0)
#     pred = rf.predict(feature)
#     label = le.inverse_transform(pred)[0]

#     with open(img_path, "rb") as img_file:
#         img_data = base64.b64encode(img_file.read()).decode("utf-8")

#     os.remove(img_path)
#     return JSONResponse(content={"prediction": label, "image": img_data})

# # Feedback submission
# @app.post("/feedback")
# async def receive_feedback(
#     prediction: str = Form(...),
#     decision: str = Form(...),
#     comment: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     feedback_id = str(uuid.uuid4())
#     image_filename = f"{feedback_id}.jpg"
#     image_path = os.path.join(feedback_image_folder, image_filename)

#     with open(image_path, "wb") as image_file:
#         image_file.write(await file.read())

#     feedback_json = {
#         "prediction": prediction,
#         "decision": decision,
#         "comment": comment,
#         "image_filename": image_filename
#     }

#     json_path = os.path.join(feedback_data_folder, f"{feedback_id}.json")
#     with open(json_path, "w") as json_file:
#         json.dump(feedback_json, json_file)

#     return JSONResponse(content={"message": "Feedback submitted successfully"})

# # Get all feedbacks
# @app.get("/feedback/all")
# def get_all_feedback():
#     feedback_list = []
#     for filename in os.listdir(feedback_data_folder):
#         if filename.endswith(".json"):
#             with open(os.path.join(feedback_data_folder, filename), "r") as f:
#                 feedback_list.append(json.load(f))
#     return feedback_list

# # Serve individual feedback image
# @app.get("/feedback/image/{image_name}")
# def get_feedback_image(image_name: str):
#     img_path = os.path.join(feedback_image_folder, image_name)
#     if os.path.exists(img_path):
#         return FileResponse(img_path)
#     return JSONResponse(status_code=404, content={"message": "Image not found"})




from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import joblib
import numpy as np
import base64
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


app = FastAPI()

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Static and template folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ✅ Load model and encoder
rf = joblib.load(os.path.join(BASE_DIR, "random_forest_dr_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_path = "temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    feature = feature_extractor.predict(img_array, verbose=0)
    pred = rf.predict(feature)
    label = le.inverse_transform(pred)[0]

    with open(img_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")

    os.remove(img_path)
    return JSONResponse(content={"prediction": label, "image": img_data})

# ✅ Feedback route (uses Supabase storage + DB)
@app.post("/feedback")
async def receive_feedback(
    prediction: str = Form(...),
    decision: str = Form(...),
    comment: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        feedback_id = str(uuid.uuid4())
        image_filename = f"{feedback_id}.jpg"
        image_data = await file.read()

        # Upload image to Supabase Storage
        response = supabase.storage.from_("feedback-images").upload(
            image_filename,
            image_data,
            {"content-type": "image/jpeg", "x-upsert": "true"}
        )

        if not response or getattr(response, "status_code", 200) >= 400:
            return JSONResponse(content={"message": "Image upload failed"}, status_code=500)

        # Construct image public URL
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/feedback-images/{image_filename}"

        # Save feedback in Supabase table
        feedback_data = {
            "id": feedback_id,
            "prediction": prediction,
            "decision": decision,
            "comment": comment,
            "image_filename": image_filename,
            "created_at": datetime.utcnow().isoformat()
        }

        db_response = supabase.table("feedbacks").insert(feedback_data).execute()
        if db_response.data is None:
            return JSONResponse(content={"message": "Failed to save feedback to database"}, status_code=500)

        return JSONResponse(content={"message": "Feedback submitted successfully"})

    except Exception as e:
        return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=500)

# ✅ Get all feedbacks
@app.get("/feedback/all")
def get_all_feedback():
    response = supabase.table("feedbacks").select("*").order("created_at", desc=True).execute()
    return response.data

# ✅ Return image URL
@app.get("/feedback/image/{image_name}")
def get_feedback_image(image_name: str):
    image_url = f"{SUPABASE_URL}/storage/v1/object/public/feedback-images/{image_name}"
    return JSONResponse(content={"image_url": image_url})
