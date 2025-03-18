import streamlit as st
import requests
import matplotlib.pyplot as plt

def plot_pie_chart(sentiment_data):
    labels = sentiment_data.keys()
    sizes = sentiment_data.values()
    colors = ['#4CAF50', '#FF5252', '#FFC107']

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)

st.title("YouTube Comment Sentiment Analysis")
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Analyze"):
    with st.spinner("Analyzing comments..."):
        try:
            response = requests.get(f"http://127.0.0.1:5000/analyze?video_url={video_url}")
            data = response.json()

            st.subheader("Sentiment Distribution")
            plot_pie_chart(data["sentiment_data"])

            st.subheader("Overall Confidence Score")
            st.write(f"Confidence: {data['average_confidence'] * 100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")