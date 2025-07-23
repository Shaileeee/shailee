from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]#as shape[0] returns the no. of rows in df

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())#splitting my message into words and storing them in words list

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]#finding media omitted in message

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()#counts the number of times a user appears in user coloumn and sort it in descending order
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})#converting to percent and then to df
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']#filtering df to remove these

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:#splitting message into words,converting it to lowercases and checking if it is
                y.append(word)#not in stopwords thn appending it to word
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))#counts the frequency of each word and return mostcommon 20 words
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
       emojis.extend([e['emoji'] for e in emoji.emoji_list(message)])#extract emoji and other info from message then
       emoji_count = Counter(emojis)#extract emoji from it and store it in emojis list
       emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()#df will be grouped by all three
           #and last coloumn will be no.of messages
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))#a new coloumn time is added to df by combining
          #month and year
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()#after grouping by onlydate date will be in
       #first coloumn count of messages on that date will be scnd
    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()#most busy day

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()#most busy month

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
       #index=row,coloumns=coloumn,values=The pivot table will count the number of messages in each combination of day_name and period.
    return user_heatmap



def perform_sentiment_analysis(selected_user, df):
    # Filter data for the selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Create a new column for sentiment polarity
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Classify the sentiment
    def classify_sentiment(polarity):
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment'].apply(classify_sentiment)

    # Aggregate the results
    sentiment_counts = df['sentiment_category'].value_counts()

    return df, sentiment_counts
