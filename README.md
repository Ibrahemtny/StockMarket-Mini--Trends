# StockMarket(Mini)-Trends
Stock Market Sentiment Analysis
This project aims to perform sentiment analysis on Twitter posts (tweets) about a given stock symbol and correlate that sentiment with stock price movements. It uses Python, Natural Language Processing (NLP) with VADER for sentiment analysis, and Linear Regression to predict stock price changes based on sentiment.

Features:
Fetch stock price data using Yahoo Finance (yfinance).
Fetch tweets mentioning a specific stock using the Twitter API.
Perform sentiment analysis on the tweets using VADER Sentiment Analysis from NLTK.
Visualize the relationship between stock prices and sentiment.
Build a Linear Regression model to predict stock price changes based on sentiment.
Requirements:
Before running the project, make sure you have the following Python libraries installed:

yfinance – To fetch stock data.
tweepy – To interact with the Twitter API.
nltk – To perform sentiment analysis using VADER.
pandas – For data manipulation and analysis.
matplotlib – To visualize the data.
sklearn – For building and evaluating a machine learning model.







Setup Instructions
Get Twitter API Credentials: To fetch tweets, you need to create a Twitter Developer account and get your API credentials. Follow these steps:

Go to the Twitter Developer Portal.
Apply for access and create a new app.
Once approved, generate the following credentials:
Consumer Key (API Key)
Consumer Secret (API Secret Key)
Access Token
