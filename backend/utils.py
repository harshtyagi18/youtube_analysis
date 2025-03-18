import requests
import re
from transformers import pipeline, AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def fetch_comments(video_id, max_comments=1000):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise Exception("API Key not found. Set YOUTUBE_API_KEY in environment variables.")
    
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "maxResults": 100,
    }

    while len(comments) < max_comments:
        response = requests.get(url, params=params)
        data = response.json()

        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break

        params["pageToken"] = data.get("nextPageToken")
        if not params.get("pageToken"):
            break

    return comments

def preprocess_comment(comment):
    comment = re.sub(r"http\S+|www\S+|https\S+", "", comment, flags=re.MULTILINE)
    comment = re.sub(r"\s+", " ", comment).strip()
    comment = re.sub(r"[^\w\s.,!?]", "", comment)
    return comment.lower()

def analyze_sentiments(comments):
    results = {"positive": 0, "negative": 0, "neutral": 0}
    total_score = 0

    for comment in comments:
        try:
            sentiment_result = sentiment_pipeline(comment[:512])
            label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']

            stars = int(label.split()[0])
            if stars >= 4:
                results["positive"] += 1
            elif stars == 3:
                results["neutral"] += 1
            else:
                results["negative"] += 1

            total_score += score
        except Exception as e:
            print(f"Error: {e}")
            results['neutral'] += 1

    average_confidence = total_score / len(comments) if comments else 0
    return results, average_confidence