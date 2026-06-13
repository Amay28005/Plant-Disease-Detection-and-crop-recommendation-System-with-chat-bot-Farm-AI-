# 🌱 FarmAI — Smart Agricultural Intelligence Platform

> An end-to-end AI-powered web platform for plant disease detection, crop recommendation, live weather advisory, and conversational farming assistance.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch) ![FastAPI](https://img.shields.io/badge/FastAPI-green?logo=fastapi) ![Accuracy](https://img.shields.io/badge/Accuracy-95.26%25-brightgreen) ![Classes](https://img.shields.io/badge/Classes-71-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

| Disease Detector | Crop Recommender | AI Assistant |
|---|---|---|
| Upload leaf → get disease + treatment | Enter soil/climate → top 3 crops | Chat with farming AI |

---

## 🧠 Deep Learning Model — Plant Disease Detector

- Trained **EfficientNet-B4** (17.6M parameters) using **PyTorch** and **timm** library on a merged dataset of **116,147 leaf images** across **71 disease classes** and **20 plant species**
- Achieved **95.26% test accuracy**, **95.31% weighted F1-score**, and **93.23% macro F1** on an 11,616-sample held-out test set
- Employed a **two-phase transfer learning strategy** — backbone frozen for warm-up epochs, followed by selective fine-tuning of the last 3 EfficientNet convolutional blocks
- Trained entirely on **NVIDIA RTX 3050 Laptop GPU (4GB VRAM)** using **Automatic Mixed Precision (AMP)** to fit within memory constraints
- Used **AdamW optimizer** with cosine annealing LR decay, label smoothing (ε=0.1), and gradient clipping (max_norm=1.0)
- **21 out of 71 classes** achieved perfect F1-score of 1.000 including Apple Black Rot, all Rice diseases, all Watermelon classes, Orange Citrus Greening, and more
- Model exported as `best_model.pth` with class names and metadata embedded in the checkpoint

---

## 🌾 Dataset

Merged **9 independent agricultural datasets** from Kaggle (CC0 Public Domain):

| Dataset | Images | Classes |
|---|---|---|
| PlantVillage | 54,303 | 38 |
| Cassava Leaf Disease | 21,400 | 5 |
| Rice Leaf Disease | 5,932 | 4 |
| Potato Leaf Disease | 4,072 | 3 |
| Plant Pathology | 2,700 | 3 |
| Sugarcane Leaf | 2,569 | 5 |
| Apple Tree Disease | 1,641 | 5 |
| Grape 400 | 1,600 | 4 |
| ESCA Dataset | 1,768 | 2 |
| **Total** | **116,147** | **71** |

- 80/10/10 train/val/test split with fixed seed=42 for reproducibility
- Augmentation pipeline: Resize → RandomCrop → Flip → Rotation → ColorJitter → ImageNet Normalize
- Handles variable input resolutions automatically via transform pipeline

---

## 🌿 Crop Recommendation System

- **Random Forest Classifier** (scikit-learn, n=200 trees) trained on UCI Crop Recommendation dataset
- Accepts **7 soil and climate parameters** as input:

| Parameter | Range |
|---|---|
| Nitrogen (N) | 0–140 kg/ha |
| Phosphorus (P) | 5–145 kg/ha |
| Potassium (K) | 5–205 kg/ha |
| Temperature | 8–45 °C |
| Humidity | 14–100 % |
| Soil pH | 3.5–9.5 |
| Rainfall | 20–300 mm |

- Returns **top-3 recommended crops** with probability scores using `predict_proba`
- Supports **22 crop categories**: Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidneybeans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon, Orange, Papaya, Pigeonpeas, Pomegranate, Rice, Watermelon

---

## 🌤️ Live Weather Integration

- Real-time weather data fetched via **OpenWeatherMap API**
- Displays current **temperature, humidity, rainfall, and weather condition** for any city worldwide
- **Auto-fill feature** — automatically populates the Crop Recommendation form with live weather data so farmers can get crop suggestions based on actual current conditions
- Helps farmers make data-driven planting decisions without manual data entry

---

## 🤖 AI Farming Assistant (Chatbot)

- Powered by **Anthropic Claude API** (`claude-sonnet-4-20250514`)
- Domain-specific system prompt constraining responses to crop management, disease treatment, soil advisory, irrigation, and pest control
- Maintains **full conversation history** within each session for contextual multi-turn responses
- Answers natural language farming queries instantly without requiring technical knowledge from the farmer

---

## 🌐 Web Platform — FarmAI

- Built with **FastAPI** (Python) backend and single-page HTML/CSS/JS frontend
- **4 integrated modules** accessible via navigation tabs:
  - 🍃 **Disease Detector** — upload leaf photo → get diagnosis + treatment
  - 🌾 **Crop Recommender** — enter soil/climate values → get top-3 crops
  - 🌤️ **Weather** — enter city → see live conditions + auto-fill crop form
  - 💬 **AI Assistant** — chat with Claude-powered farming expert
- Green agricultural theme, mobile responsive, loading spinners on all API calls

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.1+, timm, EfficientNet-B4 |
| ML | scikit-learn (Random Forest) |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML5, CSS3, JavaScript |
| AI Chatbot | Anthropic Claude API |
| Weather | OpenWeatherMap API |
| GPU Training | NVIDIA RTX 3050 (CUDA + AMP) |

---

## 📁 Project Structure

```
plant_disease/
├── config.py                        # All hyperparameters and paths
├── dataset.py                       # Data loading, augmentation, splits
├── model.py                         # EfficientNet-B4 + checkpointing
├── train.py                         # Full training loop with AMP
├── evaluate.py                      # Accuracy, F1, confusion matrix
├── predict.py                       # Single image / folder inference
├── main.py                          # FastAPI backend (all 4 modules)
├── templates/
│   └── index.html                   # Web frontend
├── outputs/
│   ├── best_model.pth               # Best checkpoint (95.26% accuracy)
│   ├── last_model.pth               # Latest epoch checkpoint
│   └── training_curves.png          # Loss/accuracy plots
└── crop_recommendation_project/
    └── models/
        └── crop_recommendation_rf.pkl
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FarmAI.git
cd FarmAI
```

### 2. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn fastapi uvicorn anthropic python-dotenv requests pillow tqdm matplotlib seaborn
```

### 3. Add API keys
Create a `.env` file in the root folder:
```
OPENWEATHER_API_KEY=your_openweathermap_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 4. Train the model (optional — use pretrained checkpoint)
```bash
python train.py
```

### 5. Evaluate the model
```bash
python evaluate.py
```

### 6. Run the web app
```bash
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

### 7. Predict on a single image
```bash
python predict.py --image path/to/leaf.jpg
python predict.py --folder path/to/test_folder/
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **95.26%** |
| Validation Accuracy | **95.19%** |
| Weighted F1-Score | **95.31%** |
| Macro F1-Score | **93.23%** |
| Macro Precision | **93.72%** |
| Macro Recall | **93.16%** |
| Total Classes | **71** |
| Training Images | **116,147** |
| Test Images | **11,616** |
| Training Epochs | **19 (early stopped)** |
| Training Hardware | **RTX 3050 Laptop GPU (4GB)** |

### Training Accuracy Progression

| Epoch | Train Acc | Val Acc | Phase |
|---|---|---|---|
| 1 | 1.48% | 1.35% | Frozen backbone — LR warmup |
| 3 | 58.94% | 67.45% | Frozen backbone — LR active |
| 5 | 80.40% | 83.17% | Frozen backbone — end |
| 6 | — | — | Backbone unfrozen |
| 19 | 94.94% | 95.11% | Best checkpoint saved |
| Final | — | **95.19%** | Early stopped |

---

## 🏥 Supported Plant Diseases (71 Classes)

Covers **20 plant species** including Apple, Bell Pepper, Blueberry, Cassava, Cherry, Coffee, Corn, Grape, Orange, Peach, Potato, Raspberry, Rice, Rose, Soybean, Squash, Strawberry, Sugarcane, Tomato, and Watermelon — both diseased and healthy classes.

---

## 📄 Research Paper

A full IEEE-format research paper documenting the complete methodology, mathematical formulations (9 equations), dataset details, and experimental results is included:

📄 `FarmAI_IEEE_Paper.pdf`

---

## 🔮 Future Work

- [ ] Mobile app deployment using Flutter
- [ ] ONNX export for offline on-device inference
- [ ] Disease severity estimation (mild / moderate / severe)
- [ ] Multilingual support (Hindi, Tamil, Telugu)
- [ ] Disease outbreak heatmap by GPS location
- [ ] Live mandi (market) crop price integration
- [ ] Government scheme finder for farmers

---

## 🙏 Acknowledgements

Thanks to the creators of PlantVillage, Cassava Leaf Disease Dataset, Rice Leaf Disease Images, and all other source datasets used in this project.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for farmers everywhere 🌱</p>
