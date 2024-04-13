import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configure the logger
logging.basicConfig(filename='logger.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    # Step 1: Data Loading
    data_df = pd.read_csv('C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/data.csv/rating.csv')
    domains_df = pd.read_csv('C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/domains_location.csv')
    traffic_df = pd.read_csv('C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/traffic_data/traffic.csv')

    # # Step 2: Data Preprocessing
    # merged_df = pd.merge(data_df, domains_df, left_on='source_name', right_on='SourceCommonName', how='left')
    # merged_df = pd.merge(merged_df, traffic_df, on='Domain', how='left')

    # Step 3: Dashboard Design
    st.sidebar.title('Dashboard Options')
    analysis_type = st.sidebar.selectbox('Select Analysis Type', ['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'])

    st.title('Websites with the Highest Count of Sentiment')

    # Step 4: Data Visualization
    # if analysis_type == 'Positive Sentiment':
    #     sentiment_counts = merged_df.groupby(['SourceCommonName', 'title_sentiment']).size().unstack(fill_value=0)
    #     top_positive = sentiment_counts['positive'].nlargest(10)
    #     st.subheader('Top Websites with the Highest Count of Positive Sentiment')
    #     st.bar_chart(top_positive)
    # elif analysis_type == 'Neutral Sentiment':
    #     sentiment_counts = merged_df.groupby(['SourceCommonName', 'title_sentiment']).size().unstack(fill_value=0)
    #     top_neutral = sentiment_counts['neutral'].nlargest(10)
    #     st.subheader('Top Websites with the Highest Count of Neutral Sentiment')
    #     st.bar_chart(top_neutral)
    # else:
    #     sentiment_counts = merged_df.groupby(['SourceCommonName', 'title_sentiment']).size().unstack(fill_value=0)
    #     top_negative = sentiment_counts['negative'].nlargest(10)
    #     st.subheader('Top Websites with the Highest Count of Negative Sentiment')
    #     st.bar_chart(top_negative)

    path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/traffic_data/traffic.csv"
    df1=pd.read_csv(path)
    df1.head(5)

    path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/data.csv/rating.csv"
    df=pd.read_csv(path)
    highest_traffic_websites = df1.nlargest(10, 'GlobalRank')
    largest_websites = df['source_name'].value_counts().nlargest(10)

    # Create a Streamlit app
    st.title('Top 10 Websites Analysis')

    # Display the first plot
    st.subheader('Top 10 Websites by Article Count')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    largest_websites.plot(kind='bar', color='skyblue', ax=ax1)
    plt.title('Top 10 Websites by Article Count')
    plt.xlabel('Website')
    plt.ylabel('Article Count')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Display the second plot
    st.subheader('Top 10 Websites by Global Rank (Visitor Traffic)')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plt.barh(highest_traffic_websites['Domain'], highest_traffic_websites['GlobalRank'], color='lightgreen')
    plt.title('Top 10 Websites by Global Rank (Visitor Traffic)')
    plt.xlabel('Global Rank')
    plt.ylabel('Website')
    plt.xscale('log')  # Use a log scale for the x-axis
    plt.gca().invert_yaxis()  # Reverse the y-axis to show the highest rank at the top
    st.pyplot(fig2)

    # Display the count of news articles for each website
    st.write(largest_websites)

    # Display the websites with the highest numbers of visitors traffic
    st.write(highest_traffic_websites)

    # Display the count of news articles for each website
    st.write(largest_websites)

except Exception as e:
    # Log the error
    logging.exception("An error occurred: %s", e)
    # Display an error message to the user
    st.error("An error occurred while processing the data. Please check the logs for more details.")
