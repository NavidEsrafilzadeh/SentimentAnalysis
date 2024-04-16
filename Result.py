from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import datetime ##


finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["AMZN", "PEP", "META"]

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url= url, headers = {'user-agent': "my-app"})
    response = urlopen(req)

    html = BeautifulSoup(response, "html")
    news_table = html.find(id = "news-table")
    news_tables[ticker] = news_table #Key and corresponding news-table for each ticker

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in  news_table.find_all("tr"): #Every news article for each ticker
        title = row.a.get_text()
        date_data = row.td.text.strip().split(" ")  #.strip() to remove any whitespace in extracted strings
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
            if date == "Today":
                 date = pd.Timestamp.today().strftime('%m-%d-%y')  # Replace "Today" with today's date

        parsed_data.append([ticker, date, time, title]) #Making a list of lists, where each list contains data for each article

df = pd.DataFrame(parsed_data, columns = ["ticker", "date", "time", "title"]) 
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)["compound"]  # only care about compound score
df["compound"] = df["title"].apply(f) #adding new column with title 'compound' in dataframe, cocnsisting of only compund scores
df["date"] = pd.to_datetime(df.date).dt.date  #converting date from string to date-time format

plt.figure(figsize= (10,8))
mean_df = df.groupby(["ticker", "date"])["compound"].mean() #Dataframe with columns ticker, date and mean of sentiment for each day, grouping all news articles for each day for the ticker
mean_df = mean_df.unstack()
mean_df = mean_df.transpose() # Transpose the DataFrame to switch rows and columns

mean_df.plot(kind = "bar")
plt.ylabel("Compound Sentiment Score")
plt.title("Sentiment Analysis by Date")
plt.show()  #run and debug to run program