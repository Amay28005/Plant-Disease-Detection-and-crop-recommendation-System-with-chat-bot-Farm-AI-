"""
main.py — FarmAI Smart Farming Assistant
FastAPI backend with 4 features:
  1. Plant Disease Detection (EfficientNet-B4)
  2. Crop Recommendation (Random Forest)
  3. Live Weather (OpenWeatherMap)
  4. AI Farming Chat (DeepSeek R1 via OpenRouter)
"""

import os
import io
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import httpx

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="FarmAI — Smart Farming Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# ── API Keys ───────────────────────────────────────────────
OWM_API_KEY = "1da824ca9cd5da9bd798d5c493e4f783"
OPENROUTER_API_KEY = "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"

# ── Constants ──────────────────────────────────────────────
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("outputs", "best_model.pth")
CROP_MODEL_PATH = os.path.join("crop_recommendation_project", "models", "crop_recommendation_rf.pkl")

# ── Disease class names (71 classes) ──────────────────────
CLASS_NAMES = [
    "Apple__alternaria_leaf_spot", "Apple__black_rot", "Apple__brown_spot",
    "Apple__gray_spot", "Apple__healthy", "Apple__rust", "Apple__scab",
    "Bell_pepper__bacterial_spot", "Bell_pepper__healthy",
    "Blueberry__healthy", "Cassava__bacterial_blight",
    "Cassava__brown_streak_disease", "Cassava__green_mottle",
    "Cassava__healthy", "Cassava__mosaic_disease", "Cherry__healthy",
    "Cherry__powdery_mildew", "Coffee__healthy", "Coffee__red_spider_mite",
    "Coffee__rust", "Corn__common_rust", "Corn__gray_leaf_spot",
    "Corn__healthy", "Corn__northern_leaf_blight", "Grape__Leaf_blight",
    "Grape__black_measles", "Grape__black_rot", "Grape__healthy",
    "Orange__citrus_greening", "Peach__bacterial_spot", "Peach__healthy",
    "Potato__bacterial_wilt", "Potato__early_blight", "Potato__healthy",
    "Potato__late_blight", "Potato__leafroll_virus", "Potato__mosaic_virus",
    "Potato__nematode", "Potato__pests", "Potato__phytophthora",
    "Raspberry__healthy", "Rice__bacterial_blight", "Rice__blast",
    "Rice__brown_spot", "Rice__tungro", "Rose__healthy", "Rose__rust",
    "Rose__slug_sawfly", "Soybean__healthy", "Squash__powdery_mildew",
    "Strawberry__healthy", "Strawberry__leaf_scorch",
    "Sugarcane__healthy", "Sugarcane__mosaic", "Sugarcane__red_rot",
    "Sugarcane__rust", "Sugarcane__yellow_leaf",
    "Tomato__bacterial_spot", "Tomato__early_blight", "Tomato__healthy",
    "Tomato__late_blight", "Tomato__leaf_curl", "Tomato__leaf_mold",
    "Tomato__mosaic_virus", "Tomato__septoria_leaf_spot",
    "Tomato__spider_mites", "Tomato__target_spot",
    "Watermelon__anthracnose", "Watermelon__downy_mildew",
    "Watermelon__healthy", "Watermelon__mosaic_virus"
]

# ── Treatment recommendations ─────────────────────────────
TREATMENTS = {
    "Apple__alternaria_leaf_spot": "Apply fungicides containing mancozeb or captan. Remove and destroy fallen leaves. Ensure good air circulation by proper pruning.",
    "Apple__black_rot": "Prune out dead or diseased branches. Apply fungicide (captan or myclobutanil) during bloom. Remove mummified fruits and cankers.",
    "Apple__brown_spot": "Apply copper-based fungicides. Improve drainage and air circulation. Remove infected leaves and debris from the orchard floor.",
    "Apple__gray_spot": "Use fungicides such as thiophanate-methyl. Maintain proper tree spacing. Remove and destroy infected plant material.",
    "Apple__healthy": "No treatment needed. Continue regular maintenance: proper watering, balanced fertilization, and annual pruning.",
    "Apple__rust": "Apply fungicides (myclobutanil or triadimefon) at early bloom. Remove nearby cedar/juniper trees (alternate hosts). Prune affected branches.",
    "Apple__scab": "Apply fungicides (captan, mancozeb) preventatively. Rake and destroy fallen leaves. Choose scab-resistant varieties for new plantings.",
    "Bell_pepper__bacterial_spot": "Apply copper-based bactericides. Use disease-free seeds. Avoid overhead irrigation. Rotate crops every 2–3 years.",
    "Bell_pepper__healthy": "No treatment needed. Maintain consistent watering, mulching, and balanced NPK fertilization.",
    "Blueberry__healthy": "No treatment needed. Maintain acidic soil pH (4.5–5.5), regular mulching, and proper pruning of old canes.",
    "Cassava__bacterial_blight": "Use disease-free planting material. Remove and burn infected plants. Apply copper-based sprays. Practice crop rotation.",
    "Cassava__brown_streak_disease": "Plant resistant varieties. Remove infected plants immediately. Use virus-free cuttings. Control whitefly vectors with insecticides.",
    "Cassava__green_mottle": "Use virus-free planting material. Remove infected plants. Control insect vectors. Practice field sanitation.",
    "Cassava__healthy": "No treatment needed. Maintain proper spacing, weed control, and balanced fertilization.",
    "Cassava__mosaic_disease": "Plant resistant/tolerant varieties. Remove and destroy infected plants. Control whitefly populations with neem-based insecticides.",
    "Cherry__healthy": "No treatment needed. Continue regular pruning, balanced fertilization, and proper irrigation management.",
    "Cherry__powdery_mildew": "Apply sulfur-based or potassium bicarbonate fungicides. Improve air circulation. Avoid excessive nitrogen fertilization.",
    "Coffee__healthy": "No treatment needed. Maintain shade management, regular pruning, and balanced fertilization with organic matter.",
    "Coffee__red_spider_mite": "Apply miticides (abamectin or spiromesifen). Increase humidity around plants. Introduce predatory mites as biological control.",
    "Coffee__rust": "Apply copper-based or systemic fungicides (triazoles). Plant resistant varieties. Ensure proper shade and spacing.",
    "Corn__common_rust": "Apply foliar fungicides (azoxystrobin or propiconazole). Plant resistant hybrids. Monitor fields during warm, humid conditions.",
    "Corn__gray_leaf_spot": "Apply strobilurin or triazole fungicides. Practice crop rotation. Use tillage to bury infected residue.",
    "Corn__healthy": "No treatment needed. Continue balanced fertilization, proper spacing, and integrated pest management.",
    "Corn__northern_leaf_blight": "Apply fungicides (propiconazole or azoxystrobin) at first symptoms. Use resistant hybrids. Rotate crops and manage residue.",
    "Grape__Leaf_blight": "Apply mancozeb or copper-based fungicides. Remove infected leaves. Ensure good canopy management for air flow.",
    "Grape__black_measles": "Prune and remove infected wood. Apply fungicide wound protectants. No fully effective chemical control—prevention is key.",
    "Grape__black_rot": "Apply fungicides (myclobutanil or mancozeb) from bud break. Remove mummified berries. Improve air circulation in canopy.",
    "Grape__healthy": "No treatment needed. Maintain balanced pruning, training, and vineyard floor management.",
    "Orange__citrus_greening": "Control Asian citrus psyllid vector with systemic insecticides. Remove and destroy infected trees. Plant certified disease-free nursery stock.",
    "Peach__bacterial_spot": "Apply copper sprays during dormancy. Use resistant varieties. Avoid overhead irrigation. Provide windbreaks to reduce injury.",
    "Peach__healthy": "No treatment needed. Continue annual pruning, thinning, and balanced fertilization schedules.",
    "Potato__bacterial_wilt": "Use certified disease-free seed tubers. Practice 3+ year crop rotation. Remove and destroy infected plants. Improve soil drainage.",
    "Potato__early_blight": "Apply fungicides (chlorothalonil or mancozeb). Practice crop rotation. Remove infected plant debris. Ensure adequate nutrition.",
    "Potato__healthy": "No treatment needed. Maintain proper hilling, irrigation management, and pest monitoring.",
    "Potato__late_blight": "Apply systemic fungicides (metalaxyl + mancozeb) preventatively. Destroy infected plants. Avoid overhead irrigation. Use resistant varieties.",
    "Potato__leafroll_virus": "Use virus-free certified seed. Control aphid vectors with insecticides. Remove infected plants promptly.",
    "Potato__mosaic_virus": "Use certified virus-free seed potatoes. Control aphid vectors. Remove and destroy symptomatic plants. Sanitize tools.",
    "Potato__nematode": "Apply nematicides or use biocontrol agents. Rotate with non-host crops (cereals). Use resistant potato varieties. Solarize soil.",
    "Potato__pests": "Apply appropriate insecticides (neem oil, spinosad). Use integrated pest management. Monitor with traps. Practice crop rotation.",
    "Potato__phytophthora": "Apply fungicides (metalaxyl, mancozeb). Improve drainage. Practice long crop rotations. Remove volunteer potato plants.",
    "Raspberry__healthy": "No treatment needed. Continue proper pruning of floricanes, weed management, and balanced fertilization.",
    "Rice__bacterial_blight": "Use resistant varieties. Apply copper-based bactericides. Ensure proper field drainage. Avoid excessive nitrogen.",
    "Rice__blast": "Apply tricyclazole or isoprothiolane fungicides. Use blast-resistant varieties. Manage nitrogen levels. Maintain consistent water depth.",
    "Rice__brown_spot": "Apply mancozeb or propiconazole fungicides. Use balanced fertilization (especially potassium). Treat seeds with fungicide before planting.",
    "Rice__tungro": "Plant resistant varieties. Control green leafhopper vectors. Synchronize planting dates. Remove infected plants and ratoon growth.",
    "Rose__healthy": "No treatment needed. Continue regular pruning, deadheading, balanced fertilization, and adequate watering.",
    "Rose__rust": "Apply myclobutanil or triforine fungicides. Remove and destroy infected leaves. Improve air circulation. Avoid wetting foliage.",
    "Rose__slug_sawfly": "Apply insecticidal soap or spinosad. Handpick larvae. Encourage natural predators. Spray neem oil as preventive measure.",
    "Soybean__healthy": "No treatment needed. Continue integrated pest management, proper inoculation, and balanced soil fertility.",
    "Squash__powdery_mildew": "Apply sulfur-based or potassium bicarbonate fungicides. Improve spacing for air flow. Use resistant varieties. Avoid overhead watering.",
    "Strawberry__healthy": "No treatment needed. Maintain proper runner management, mulching, and renovation after harvest.",
    "Strawberry__leaf_scorch": "Apply fungicides (captan or thiram). Remove old infected leaves. Improve air circulation. Ensure proper spacing between plants.",
    "Sugarcane__healthy": "No treatment needed. Maintain proper irrigation, ratoon management, and balanced NPK fertilization.",
    "Sugarcane__mosaic": "Plant virus-free setts from resistant varieties. Remove infected stools. Control aphid vectors with insecticides.",
    "Sugarcane__red_rot": "Use disease-free seed cane. Plant resistant varieties. Apply fungicide dip to setts before planting. Remove infected clumps.",
    "Sugarcane__rust": "Plant resistant varieties. Apply propiconazole or mancozeb fungicides. Ensure proper spacing and nutrition to reduce susceptibility.",
    "Sugarcane__yellow_leaf": "Use virus-free planting material. Control aphid vectors. Remove symptomatic plants. Plant resistant varieties.",
    "Tomato__bacterial_spot": "Apply copper-based bactericides. Use disease-free seeds/transplants. Avoid overhead irrigation. Rotate crops every 2–3 years.",
    "Tomato__early_blight": "Apply chlorothalonil or mancozeb fungicides. Mulch around base. Remove lower affected leaves. Rotate crops annually.",
    "Tomato__healthy": "No treatment needed. Continue staking, pruning suckers, regular watering, and balanced calcium fertilization.",
    "Tomato__late_blight": "Apply metalaxyl + mancozeb immediately. Remove and destroy all infected tissue. Avoid overhead watering. Use resistant varieties.",
    "Tomato__leaf_curl": "Control whitefly vectors with imidacloprid or neem oil. Use reflective mulches. Plant resistant varieties. Remove infected plants.",
    "Tomato__leaf_mold": "Improve greenhouse ventilation. Apply chlorothalonil or copper fungicides. Reduce humidity. Use resistant varieties.",
    "Tomato__mosaic_virus": "Remove and destroy infected plants. Sanitize tools with 10% bleach. Use resistant varieties. Avoid tobacco products near plants.",
    "Tomato__septoria_leaf_spot": "Apply chlorothalonil or mancozeb fungicides. Remove infected lower leaves. Mulch to prevent soil splash. Rotate crops.",
    "Tomato__spider_mites": "Apply miticides (abamectin) or insecticidal soap. Increase humidity. Introduce predatory mites. Avoid dusty conditions.",
    "Tomato__target_spot": "Apply chlorothalonil or azoxystrobin fungicides. Remove infected leaves. Improve air circulation. Practice crop rotation.",
    "Watermelon__anthracnose": "Apply chlorothalonil or mancozeb fungicides. Use disease-free seeds. Practice 2-year crop rotation. Avoid overhead irrigation.",
    "Watermelon__downy_mildew": "Apply metalaxyl-based or mandipropamid fungicides. Ensure good air flow. Remove infected leaves. Monitor during humid weather.",
    "Watermelon__healthy": "No treatment needed. Continue proper vine management, drip irrigation, and balanced fertilization.",
    "Watermelon__mosaic_virus": "Control aphid vectors with reflective mulch or insecticides. Use resistant varieties. Remove and destroy infected plants promptly."
}

# ── Load plant disease model ──────────────────────────────
import timm

def load_disease_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"[FarmAI] Disease model loaded — {num_classes} classes on {DEVICE}")
    return model, class_names

disease_model, disease_class_names = load_disease_model()

# ── Load crop recommendation model ────────────────────────
with open(CROP_MODEL_PATH, "rb") as f:
    crop_model = pickle.load(f)
print(f"[FarmAI] Crop model loaded — classes: {list(crop_model.classes_)}")

# ── Image transform ───────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Pydantic models ───────────────────────────────────────
class CropInput(BaseModel):
    N: float = Field(..., ge=0, le=140, description="Nitrogen")
    P: float = Field(..., ge=5, le=145, description="Phosphorus")
    K: float = Field(..., ge=5, le=205, description="Potassium")
    temperature: float = Field(..., ge=8, le=45, description="Temperature °C")
    humidity: float = Field(..., ge=14, le=100, description="Humidity %")
    ph: float = Field(..., ge=3.5, le=9.5, description="Soil pH")
    rainfall: float = Field(..., ge=20, le=300, description="Rainfall mm")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    message: str
    history: List[ChatMessage] = []

# ══════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = disease_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idxs = probs.topk(min(5, len(disease_class_names)))
    predictions = []
    for idx, prob in zip(top5_idxs, top5_probs):
        name = disease_class_names[idx.item()]
        confidence = round(prob.item() * 100, 2)
        treatment = TREATMENTS.get(name, "Consult a local agricultural extension officer for specific treatment advice.")
        predictions.append({
            "disease": name,
            "confidence": confidence,
            "treatment": treatment
        })

    return {
        "success": True,
        "prediction": predictions[0]["disease"],
        "confidence": predictions[0]["confidence"],
        "top5": predictions
    }


@app.post("/recommend-crop")
async def recommend_crop(data: CropInput):
    features = np.array([[data.N, data.P, data.K, data.temperature,
                          data.humidity, data.ph, data.rainfall]])
    probas = crop_model.predict_proba(features)[0]
    top3_idxs = np.argsort(probas)[-3:][::-1]

    recommendations = []
    for idx in top3_idxs:
        recommendations.append({
            "crop": crop_model.classes_[idx],
            "confidence": round(probas[idx] * 100, 2)
        })

    return {"success": True, "recommendations": recommendations}


@app.get("/weather/{city}")
async def get_weather(city: str):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OWM_API_KEY, "units": "metric"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        detail = resp.json().get("message", "City not found")
        raise HTTPException(status_code=resp.status_code, detail=detail)

    data = resp.json()
    rain = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0)

    return {
        "success": True,
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": round(data["main"]["temp"], 1),
        "humidity": data["main"]["humidity"],
        "rainfall": round(rain, 1),
        "condition": data["weather"][0]["description"].title(),
        "icon": data["weather"][0]["icon"],
        "wind_speed": round(data.get("wind", {}).get("speed", 0), 1),
        "feels_like": round(data["main"]["feels_like"], 1),
    }


@app.post("/chat")
async def chat(data: ChatInput):
    system_prompt = (
        "You are FarmAI, an expert agricultural assistant. "
        "You help farmers with plant diseases, crop selection, "
        "soil management, irrigation, and pest control. "
        "Keep answers concise and practical. "
        "When a disease is mentioned, always suggest treatment."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in data.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": data.message})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "FarmAI Smart Farming Assistant",
    }

    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="AI service error — please try again")

    result = resp.json()
    reply = result["choices"][0]["message"]["content"]
    # Strip <think>...</think> tags from DeepSeek R1 reasoning
    import re
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

    return {"success": True, "reply": reply}
