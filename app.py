from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastai.text.all import load_learner
from newspaper import Article

app = FastAPI()

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
        return {"label": str(pred[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
