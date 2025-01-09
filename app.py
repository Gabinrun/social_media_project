import streamlit as st
import pandas as pd
import os
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import pytz


# Load CSV data
def load_data(user_file):
    file_path = os.path.join("data", user_file)
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Perform sentiment analysis
def analyze_sentiments(df):
    sentiments = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = sentiments
    return df

# Generate word cloud with cleaned content
def generate_wordcloud(text):
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["https", "co", "t"])  # Exclude common URL words
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                          stopwords=custom_stopwords).generate(' '.join(text))
    return wordcloud

# Plot the evolution of tweets over time (last 3 months)
def plot_tweets_over_time(df):
    # Définir la date limite (3 mois en arrière) avec fuseau horaire UTC
    three_months_ago = (datetime.now() - timedelta(days=90)).astimezone(pytz.UTC)
    recent_tweets = df[df['date'] >= three_months_ago]
    tweets_per_day = recent_tweets.groupby(recent_tweets['date'].dt.date).size()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=tweets_per_day.index, y=tweets_per_day.values, marker="o", color="#2A9D8F")
    plt.title("Tweet Activity (Last 3 Months)", fontsize=20, color="#264653")
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Number of Tweets", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt

# Dashboard structure and interactivity
def main():
    st.set_page_config(page_title="Twitter Dashboard", layout="wide")
    st.title("📊 Twitter Insights Dashboard")
    st.markdown("#### Gabin Garrot, support for the social media project - TU Wien 2025")
    st.markdown("Please, select a user to explore user tweet data, perform sentiment analysis, and uncover key trends.")
    
    st.sidebar.header("User Selection")
    csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    users = [os.path.splitext(f)[0] for f in csv_files]
    selected_user = st.sidebar.selectbox("Choose a Twitter User", users)
    
    if selected_user:
        st.subheader(f"Dashboard for **{selected_user}**")
        user_file = f"{selected_user}.csv"
        df = load_data(user_file)

        st.write(f"### Latest Tweets from {selected_user}")
        st.dataframe(df[['date', 'content', 'likes', 'retweets', 'replies']].head(10))

        # Key metrics section
        st.markdown("### Key Metrics")
        total_tweets = len(df)
        avg_likes = df['likes'].mean()
        avg_retweets = df['retweets'].mean()
        avg_replies = df['replies'].mean()
        std_likes = np.std(df['likes'])
        std_retweets = np.std(df['retweets'])
        std_replies = np.std(df['replies'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", total_tweets)
        col2.metric("Avg Likes", f"{avg_likes:.1f}")
        col3.metric("Avg Retweets", f"{avg_retweets:.1f}")
        col4.metric("Avg Replies", f"{avg_replies:.1f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Likes Std Dev", f"{std_likes:.1f}")
        col6.metric("Retweets Std Dev", f"{std_retweets:.1f}")
        col7.metric("Replies Std Dev", f"{std_replies:.1f}")

        # Sentiment distribution
        df = analyze_sentiments(df)
        st.markdown("### Sentiment Distribution")
        st.markdown("This chart shows the polarity of tweets, where **-1 is Negative** and **1 is Positive**.")
        st.bar_chart(df['sentiment'], use_container_width=True)

        # Word cloud visualization
        st.markdown("### Word Cloud of Tweet Content")
        st.markdown("A visualization of the most frequently used words, excluding common URL fragments.")
        wordcloud = generate_wordcloud(df['content'])
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Plot tweet activity
        st.markdown("### Tweet Activity Over Time")
        st.markdown("Analyze the frequency of tweets to uncover trends and identify high-activity periods.")
        tweet_time_plot = plot_tweets_over_time(df)
        st.pyplot(tweet_time_plot)

        # Highlight top-performing tweets
        st.markdown("### Top Performing Tweets")
        most_liked_tweet = df.loc[df['likes'].idxmax()]
        st.write(f"**Most Liked Tweet:** {most_liked_tweet['content']} \n(Likes: {most_liked_tweet['likes']})")
        most_retweeted_tweet = df.loc[df['retweets'].idxmax()]
        st.write(f"**Most Retweeted Tweet:** {most_retweeted_tweet['content']} \n(Retweets: {most_retweeted_tweet['retweets']})")

        # Insights and observations
        st.markdown("### Insights & Observations")
        st.markdown("""
        This data reflects the last 5,000 to 7,000 tweets and retweets from the selected user. Analyses and criticisms are provided directly in the report linked to the project, so please refer to it.
        """)

if __name__ == "__main__":
    main()
