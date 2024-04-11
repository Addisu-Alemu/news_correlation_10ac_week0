import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/data.csv/rating.csv"
# df=pd.read_csv(path)
# df.head(5)

# Take the top 300 rows
# top_300 = df.head(300)

# # Save the top 300 rows to a new CSV file
# top_300.to_csv('top_300_data.csv', index=False)

path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/top_300_data.csv"
df_top=pd.read_csv(path)

# path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/traffic_data/traffic.csv"
# df1=pd.read_csv(path)
# df1.head(5)

# path="C:/Users/addisu/Documents/GitHub/news_correlation_10ac_week0/domains_location.csv"
# df2=pd.read_csv(path)
# df2.head(5)

#  1. Websites that have the largest count of news articles

# largest_websites = df['source_name'].value_counts().nlargest(10)
# print(largest_websites)

  

# 2.Websites with the highest numbers of visitors traffic 
# highest_traffic_websites = df1.nlargest(10, 'GlobalRank')
# print(highest_traffic_websites) 
     
# GlobalRank	TldRank	Domain	TLD	RefSubNets	RefIPs	IDN_Domain	IDN_TLD	PrevGlobalRank	PrevTldRank	PrevRefSubNets	PrevRefIPs
# 999999	1000000	485328	toyotamusicfactory.com	com	222	280	toyotamusicfactory.com	com	973201	471949	228	284
# 999998	999999	485327	soderhomes.com	com	222	280	soderhomes.com	com	-1	-1	-1	-1
# 999997	999998	485326	pinkwater.com	com	222	280	pinkwater.com	com	-1	-1	-1	-1
# 999996	999997	485325	mt-lock.com	com	222	280	mt-lock.com	com	952633	461429	232	284
# 999995	999996	485324	kireie.com	com	222	280	kireie.com	com	-1	-1	-1	-1
# 999994	999995	485323	keith-baker.com	com	222	280	keith-baker.com	com	973079	471891	228	287
# 999993	999994	485322	irishcycle.com	com	222	280	irishcycle.com	com	968062	469365	229	285
# 999992	999993	485321	hmag.com	com	222	280	hmag.com	com	-1	-1	-1	-1
# 999991	999992	485320	exploring-africa.com	com	222	280	exploring-africa.com	com	-1	-1	-1	-1
# 999990	999991	485319	eiretrip.com	com	222	280	eiretrip.com	com	-1	-1	-1	-1

# 3.Countries with the highest number of news media organisations (represented by domains in the data)
# highest_news_media_organisations = df2['Country'].value_counts().nlargest(10)
# print(highest_news_media_organisations)
#      Country
# United States     14111
# United Kingdom     1950
# Italy              1810
# France             1041
# Russia             1024
# Canada              887
# Germany             884
# China               780
# Turkey              725
# India               686
# Name: count, dtype: int64


# 4.Countries that have many articles written about them - the content of the news is about that country
# countries_with_many_articles = df['category'].value_counts().nlargest(10)
# print(countries_with_many_articles)

# category
# Stock          3687
# Canada         2066
# Health         2046
# Real estate    2030
# Technology     1993
# Finance        1850
# News           1401
# COVID          1345
# Education      1325
# Food           1144
# Name: count, dtype: int64

# countries_with_many_articles = df['category'].value_counts().nsmallest(10)
# print(countries_with_many_articles)
#  ategory
# Martinique    2
# Réunion       2
# Andorra       3
# Burundi       3
# Cabo Verde    3
# Honduras      3
# San Marino    4
# Gambia        4
# Minimalism    4
# Viet Nam      6
# Name: count, dtype: int64

# 5.Websites that reported (the news content) about Africa, US, China, EU, Russia, Ukraine, Middle East? Note that you will need to group countries together to form the African, EU, and Middle East continents/regions.



# # Create a dictionary to map countries to their corresponding regions


#  [ ] first df['category'].unique( )   / to know the listsin the category and then to insert in their corresponding region like 'south sudan' to africa

# country_to_region = {
#     'Africa': ['Madagascar','Mali'],  # Add more African countries as needed
#     'US': ['United States', 'USA'],  # Add more US-related keywords as needed
#     'China': ['China'],  # Add more Chinese-related keywords as needed
#     'EU': ['European Union', 'Germany', 'France', 'UK'],  # Add more EU-related keywords as needed
#     'Russia': ['Russia'],  # Add more Russian-related keywords as needed
#     'Ukraine': ['Ukraine'],  # Add more Ukrainian-related keywords as needed
#     'Middle East': ['Middle East', 'Saudi Arabia', 'Iran']  # Add more Middle Eastern countries as needed
# }

# # Create a new column for each region indicating if it's mentioned in the content
# for region, countries in country_to_region.items():
#     df[region] = df['content'].str.contains('|'.join(countries), case=False)

# # Count the occurrences of each region across all articles
# region_counts = df[list(country_to_region.keys())].sum()

# print("Number of articles mentioning each region:")
# print(region_counts)

# Africa          155
# US             1799
# China          1000
# EU             2637
# Russia         1312
# Ukraine         868
# Middle East    1035
# dtype: int64

# Handling Missing Values
columns_for_missing_values = ['article_id', 'source_id', 'source_name', 'author', 'title', 'description', 'url_to_image', 'published_at', 'content', 'category', 'article', 'title_sentiment']
df_top.dropna(subset=columns_for_missing_values, inplace=True)  # Remove rows with missing values in the specified columns

# Removing Duplicates
columns_for_duplicates = ['article_id', 'source_id', 'source_name', 'author', 'title', 'description', 'url_to_image', 'published_at', 'content', 'category', 'article', 'title_sentiment']
df_top.drop_duplicates(subset=columns_for_duplicates, inplace=True)  # Remove duplicate rows based on the specified columns

# Text Preprocessing
df_top['content'] = df_top['content'].str.lower()  # Convert content to lowercase

# Save the cleaned data to a new CSV file
df_top.to_csv('cleaned_data_top.csv', index=False)

# # Handling Missing Values
# df_top.dropna(subset=['column_name'], inplace=True)  # Remove rows with missing values in a specific column

# # Removing Duplicates
# df_top.drop_duplicates(subset=['column_name'], inplace=True)  # Remove duplicate rows based on specific columns

# # Text Preprocessing
# df_top['text_column'] = df_top['text_column'].str.lower()  # Convert text to lowercase

# # Save the cleaned data to a new CSV file
# df_top.to_csv('cleaned_data_top.csv', index=False)


# # Preprocessing the text data
# # For example, you can combine 'title' and 'description' into a single column 'text'
# df_top['text'] = df_top['title'] + ' ' + df_top['description']

# # Initialize the TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# # Fit and transform the text data
# tfidf_matrix = tfidf_vectorizer.fit_transform(df_top['text'])

# # Get the feature names (i.e., the keywords)
# feature_names = tfidf_vectorizer.get_feature_names_out()

# # Print the feature names
# print(feature_names)

# # Preprocessing the text data
# df_top['combined_text'] = df_top['title'] + ' ' + df_top['content']

# # Initialize the TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# # Fit and transform the combined text data
# tfidf_matrix = tfidf_vectorizer.fit_transform(df_top['combined_text'])

# # Calculate cosine similarity between the TF-IDF matrices of title and content
# similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # Assuming the first half of the rows correspond to titles and the second half correspond to content
# title_similarities = similarity_matrix[:len(df_top)//2, :len(df_top)//2]
# content_similarities = similarity_matrix[len(df_top)//2:, len(df_top)//2:]

# # Print the similarities
# print("Similarity between titles:")
# print(title_similarities)

# print("Similarity between content:")
# print(content_similarities)






