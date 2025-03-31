from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastai.text.all import load_learner
from newspaper import Article
import os

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the index.html file
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Load the exported FastAI model
learn = load_learner("fake_news_classifier.pkl")

class ArticleURL(BaseModel):
    url: str

@app.post("/predict")
def predict_article(data: ArticleURL):
    try:
        article = Article(data.url)
        article.download()
        article.parse()
        text = article.text

        if not text or len(text) < 20:
            raise ValueError("Article text is too short or could not be parsed.")

        pred = learn.predict(text)
        return {
            "label": str(pred[0]),
            "confidence": float(max(pred[2]))  # probability of predicted class
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

