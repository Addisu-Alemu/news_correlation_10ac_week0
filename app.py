import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Set the title of the page
st.title('News Correlation Dashboard')



path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/data.csv/rating.csv"
df=pd.read_csv(path)
largest_websites = df['source_name'].value_counts().nlargest(10)
print(largest_websites)

# Plot the count of news articles for each website
plt.figure(figsize=(10, 6))
largest_websites.plot(kind='bar', color='skyblue')
plt.title('Top 10 Websites by Article Count')
plt.xlabel('Website')
plt.ylabel('Article Count')
plt.xticks(rotation=45)
plt.show()

