import yfinance as yf
import pandas as pd
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Twitter API credentials (replace these with your own credentials)
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with the Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Function to fetch stock price data for the past month
def fetch_stock_data(symbol='TSLA'):
    stock_data = yf.download(symbol, start="2024-12-01", end=datetime.today().strftime('%Y-%m-%d'))
    return stock_data

# Example: Fetch Tesla stock data
stock_data = fetch_stock_data('TSLA')
print(stock_data.head())

# Function to fetch tweets mentioning the stock symbol
def fetch_tweets(query='TSLA', count=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang='en', tweet_mode='extended').items(count):
        tweets.append(tweet.full_text)
    return tweets

# Get some tweets about TSLA
tweets = fetch_tweets('TSLA', 100)


# Get some tweets about TSLA
tweets = fetch_tweets('TSLA', 100)

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment score for a list of texts
def get_sentiment_score(texts):
    sentiment_scores = []
    for text in texts:
        sentiment_score = sid.polarity_scores(text)
        sentiment_scores.append(sentiment_score['compound'])  # We use the 'compound' score
    return sentiment_scores

# Analyze the sentiment of the fetched tweets
sentiment_scores = get_sentiment_score(tweets)

# Print some sentiment scores
print(sentiment_scores[:10])

# Calculate average sentiment per day (assuming you want to use tweet date information later)
tweet_dates = [tweet.created_at.date() for tweet in fetch_tweets('TSLA', 100)]  # assuming we fetch dates
sentiment_df = pd.DataFrame({'Date': tweet_dates, 'Sentiment': sentiment_scores})
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# Aggregate sentiment by date (average sentiment per day)
daily_sentiment = sentiment_df.groupby('Date').mean()

# Now merge with stock price data (using close price as a simple indicator)
stock_data['Date'] = stock_data.index
merged_data = pd.merge(stock_data[['Date', 'Close']], daily_sentiment, on='Date', how='inner')

print(merged_data.head())

# Plot stock price and sentiment on the same graph
plt.figure(figsize=(10, 6))

# Plot Stock Price
plt.plot(merged_data['Date'], merged_data['Close'], label='Stock Price', color='blue')

# Plot Sentiment Score
plt.plot(merged_data['Date'], merged_data['Sentiment'], label='Sentiment', color='red', linestyle='--')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Stock Price vs Sentiment')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Preparing data for regression (use Sentiment to predict stock price movement)
merged_data['Price_Change'] = merged_data['Close'].pct_change().shift(-1)  # Predict next day's price change

# Drop the NaN values (due to shift operation)
merged_data = merged_data.dropna()

# Define features and target
X = merged_data[['Sentiment']]
y = merged_data['Price_Change']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock price change
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted price changes
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price Change')
plt.ylabel('Predicted Price Change')
plt.title('Actual vs Predicted Stock Price Change')
plt.show()

