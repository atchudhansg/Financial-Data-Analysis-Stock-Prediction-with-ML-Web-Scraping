# Financial Data Analysis & Stock Prediction

This project integrates financial data analysis, stock price prediction using machine learning, sentiment analysis on market news headlines from LiveMint via web scraping, and investment risk assessment based on predictive models.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

### Stock Price Prediction

- **Objective:** Predict closing prices of HDB stock based on historical data and sentiment scores.
- **Techniques Used:**
  - **Random Forest Regressor:** Machine learning algorithm for regression tasks.
  - **Data Preparation:** Cleaning and preprocessing of financial data (`result1.csv`).
  - **Evaluation:** Mean Squared Error (MSE) calculation to assess prediction accuracy.
  - **Visualization:** Plotting of predicted vs. actual values using matplotlib.

### Web Scraping & Sentiment Analysis

- **Objective:** Extract market news headlines from LiveMint and analyze sentiment using transformer models.
- **Technologies & Libraries:**
  - **Web Scraping:** BeautifulSoup for parsing HTML and extracting news headlines from LiveMint's market section (`stock_analysis.py`).
  - **Sentiment Analysis:** Transformer model (`cardiffnlp/twitter-roberta-base-sentiment`) for sentiment classification of headlines and news items.
  - **Preprocessing:** Handling text preprocessing to improve sentiment analysis accuracy, including user mentions and URLs.
  - **Output:** Printing sentiment labels (Negative, Neutral, Positive) and scores for each headline and news item.

### Investment Risk Assessment

- **Objective:** Evaluate investment risks based on predicted stock prices, historical data, and financial metrics.
- **Tools & Methods:**
  - **Jupyter Notebook (`finalmodel.ipynb`):** Detailed analysis using Python code blocks.
  - **Metrics:** Assessing expected return and historical volatility to make investment decisions.
  - **Function:** Defined function (`assess_risks`) to automate risk assessment based on financial metrics like P/E ratio and historical returns.

## Installation

1. **Clone the repository:**
   ```sh
   https://github.com/atchudhansg/Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping.git
   cd Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping

2. **Installing Dependencies:**
   
   pip install -r requirements.txt

4. Place the result1.csv file in the root directory of the project.

Usage:

1. Run the main script (main.py) to preprocess data, train the model, and make predictions:

   python main.py

   - The script performs stock price prediction using a Random Forest Regressor and evaluates its performance with Mean Squared Error (MSE). It plots a comparison of predicted and
      actual values using matplotlib.
   - Execute stock_analysis.py to perform web scraping and sentiment analysis on LiveMint market news:
     
     python stock_analysis.py
   - The script scrapes market news headlines from LiveMint and analyzes sentiment using a pre-trained transformer model (cardiffnlp/twitter-roberta-base-sentiment). It prints sentiment labels and scores for each headline and news item.
   - Open finalmodel.ipynb for detailed risk assessment using the trained model. The notebook assesses investment risks based on predicted prices, historical data, and financial metrics. It utilizes a defined function for risk assessment based on expected return and historical volatility.

License

This project is licensed under the MIT License. See the LICENSE file for details.

