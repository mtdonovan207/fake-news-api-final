from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastai.learner import load_learner
from newspaper import Article
import gdown
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define the model path and the Google Drive URL
MODEL_PATH = "fake_news_classifier.pkl"
GDRIVE_URL = "https://drive.google.com/uc?id=1Y6vwEQomlYdREOWZDoIO5a3CNUwozyIN"

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
learn = load_learner(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, url: str = Form(...)):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if not text or len(text) < 20:
            raise ValueError("Article text is too short or could not be parsed.")

        pred = learn.predict(text)
        label = str(pred[0])
        confidence = float(pred[2].max()) * 100

        return templates.TemplateResponse("index.html", {
            "request": request,
            "label": label,
            "confidence": f"{confidence:.2f}",
            "url": url
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "url": url
        })


