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

# ✅ App and CORS setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Use ["https://yourfrontend.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ✅ Load model and encoder
rf = joblib.load(os.path.join(BASE_DIR, "random_forest_dr_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Home page route (also handles HEAD)
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
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
