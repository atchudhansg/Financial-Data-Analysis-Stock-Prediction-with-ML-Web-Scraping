import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# URL of the website
url = "https://www.livemint.com/market"

# Send a GET request to the website
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all <li> elements with the class "newsBlock"
    news_blocks = soup.find_all("li", class_="newsBlock")

    # Extract headlines from each <li> element
    headlines = []
    for news_block in news_blocks:
        # Find the <h2> element within the <li> element
        headline_element = news_block.find("h2")

        # Extract the text content of the <h2> element
        if headline_element:
            headline = headline_element.text.strip()
            headlines.append(headline)

    # Find all <h3> elements within the specified <div> element
    h3_elements = soup.select(".market-new-common-collection_contentBox__leEBU h3")

    # Extract data from the first 7 <a> tags within <h3> elements
    stock_market_news = []
    if h3_elements:
        for i, h3_element in enumerate(h3_elements):
            if i == 7:
                break
            a_tag = h3_element.find("a")
            if a_tag:
                news_text = a_tag.text.strip()
                stock_market_news.append(news_text)
            else:
                print("No <a> tag found under the <h3> element.")
    else:
        print("No <h3> elements found.")

    # Print the headlines
    print("\nMarket news:")
    for headline in headlines:
        # Sentiment Analysis
        # load model and tokenizer
        roberta = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(roberta)
        labels = ['Negative', 'Neutral', 'Positive']

        # preprocess headline
        headline_words = []

        for word in headline.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            headline_words.append(word)

        headline_proc = " ".join(headline_words)

        # sentiment analysis
        encoded_headline = tokenizer(headline_proc, return_tensors='pt')
        output = model(**encoded_headline)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        max_score_index = scores.argmax()
        sentiment_label = labels[max_score_index]
        sentiment_score = scores[max_score_index]

        print("Headline:", headline)
        print("Sentiment:", sentiment_label, sentiment_score)

    # Print the stock market news
    print("\nStock Market News:")
    for news in stock_market_news:
        # Sentiment Analysis
        # preprocess news
        news_words = []

        for word in news.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            news_words.append(word)

        news_proc = " ".join(news_words)

        # sentiment analysis
        encoded_news = tokenizer(news_proc, return_tensors='pt')
        output = model(**encoded_news)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        max_score_index = scores.argmax()
        sentiment_label = labels[max_score_index]
        sentiment_score = scores[max_score_index]

        print("News:", news)
        print("Sentiment:", sentiment_label, sentiment_score)

else:
    print("Failed to retrieve webpage:", response.status_code)
