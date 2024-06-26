# Financial Data Analysis & Stock Prediction with Machine Learning and Web Scraping

This project integrates financial data analysis, stock price prediction using machine learning, sentiment analysis on market news headlines from LiveMint via web scraping, and investment risk assessment based on predictive models.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How This Project is Useful](#how-this-project-is-useful)
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
   git clone https://github.com/atchudhansg/Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping.git
   cd Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping


2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
3. Place the result1.csv file in the root directory of the project.

## Usage:

1. Run the main script (main.py) to preprocess data, train the model, and make predictions:
   ```sh
   python main.py
   
 - The script performs stock price prediction using a Random Forest Regressor and evaluates its performance with Mean Squared Error (MSE). It plots a comparison of predicted and
    actual values using matplotlib.
 - Execute stock_analysis.py to perform web scraping and sentiment analysis on LiveMint market news:
   ```sh
   python Stock Analysis.py
 - The script scrapes market news headlines from LiveMint and analyzes sentiment using a pre-trained transformer model (cardiffnlp/twitter-roberta-base-sentiment). It prints sentiment labels and scores for each headline and news item.
 - Open finalmodel.ipynb for detailed risk assessment using the trained model. The notebook assesses investment risks based on predicted prices, historical data, and financial metrics. It utilizes a defined function for risk assessment based on expected return and historical volatility.

## How This Project is Useful

This project serves several practical purposes:

- Investment Decision Support: By predicting stock prices and assessing investment risks, investors can make informed decisions on buying, selling, or holding stocks.
- Market Sentiment Analysis: Analyzing market sentiment from news headlines helps in understanding public perception and potential market trends.
- Educational Purposes: The project demonstrates practical applications of machine learning in finance, including data preprocessing, model training, evaluation, and visualization.
- Automation: Automated scripts for data scraping and analysis reduce manual effort and streamline information gathering.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## How This Project is Useful

This project serves several practical purposes:

- **Investment Decision Support:** By predicting stock prices and assessing investment risks, investors can make informed decisions on buying, selling, or holding stocks.
- **Market Sentiment Analysis:** Analyzing market sentiment from news headlines helps in understanding public perception and potential market trends.
- **Educational Purposes:** The project demonstrates practical applications of machine learning in finance, including data preprocessing, model training, evaluation, and visualization.
- **Automation:** Automated scripts for data scraping and analysis reduce manual effort and streamline information gathering.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Screenshot of Stock Analysis.py:

![Screenshot](https://github.com/atchudhansg/Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping/assets/116624804/9771c19b-4560-4ee8-a199-7a8457298d79)


## Yahoo Finance Data Scraping

To fetch historical data from Yahoo Finance, use the following code snippet:

The 'ticker' variable contains the name of stock as mentioend in Yahoo Finance, and the output string returns a link from where the dataset (`result1.csv`) can be downloaded.

```python
import time
import datetime

ticker = 'INR=X'
period1 = int(time.mktime(datetime.datetime(2003, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2020, 4, 28, 23, 59).timetuple()))
interval = '1d'
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true;'
print(query_string)


