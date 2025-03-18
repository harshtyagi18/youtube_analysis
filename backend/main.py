from fastapi import FastAPI, HTTPException
from .utils import fetch_comments, analyze_sentiments

app = FastAPI()

@app.get("/analyze")
def analyze_comments(video_url: str):
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        comments = fetch_comments(video_id)
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found")

        sentiment_data, average_confidence = analyze_sentiments(comments)
        return {"sentiment_data": sentiment_data, "average_confidence": average_confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
