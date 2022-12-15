# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:46:21 2022

@author: Tim Gruber, Domenico Maffei, Mason Tjuanta, Michael Birkenzeller

"""
#%% Import all the libraries


# For the missing libraries use: pip install
from __future__ import print_function
import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
import datetime as dt
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords


#%% FUNCTIONS

def clean_text(text):
    """
    Text cleaning process for the sentiment analysis 
    (for the wordcloud use clean_data function additionaly)
    Parameters
    ----------
    text : str
        TAKES IN A TEXT IN STR FORM.

    Returns
    -------
    text : str
        RETURNS THE TEXT WITHOUT UNNECCESARY CHARACTERS, IN LOWER CASE AND READY FOR A SENTIMENT ANALYSIS.
    """
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = text.strip(" ")
    text = re.sub(r"[^\w\s]", "", text)
    return text

def cut_sentence(text):
    """
    Cuts a text into seperate sentences, which are savedin a dataframe.
    Parameters
    ----------
    text : str
        TAKES IN A TEXT IN STR FORM.

    Returns
    -------
    TYPE : pd.DataFrame
        RETURNS A DF WITH ONE COLUMN 
        - "Sentences"    each row of the column represents one sentence of the text supplyed to the funciton
    """
    df = (
    pd.DataFrame(text.split("."))
    .stack()
    .reset_index()
    .rename(columns={0: "Sentences"})
    .drop("level_0", axis=1)
    .drop("level_1", axis=1)
    )
    df["Sentences"] = df["Sentences"].str.strip()
    df = df[df["Sentences"].astype(bool)].reset_index(drop=True)
    return df

def create_sentiment(df):
    """
    Parameters
    ----------
    df : pd.DataFrame
        TAKES IN A DATAFRME, HERE: THE CREATED DATA FRAME BY THE CUT_SENTENCES FUNCTION.

    Returns
    -------
    df : pd.DataFrame
        RETURNS THE SAME DATAFRAME BUT ADDS 4 COLUMNS 
        - "Clean_Text"      the same sentences as in "Sentences" but after applying the clean_text function
        - "Sentiment"       a dict containing the Sentiment Label Positive or Negative and the corresponding score
        - "Sentiment_Label" the label of the previously stated dict
        - "Sentiment_Score" the Sentiment Score of the previous dict (but from -1 to 1 where minus values represent neg. scores)

    """
    df["Clean_Text"] = df['Sentences'].map(lambda text: clean_text(text))
    corpus = list(df['Clean_Text'].values)
    df["Sentiment"] = nlp_sentiment(corpus)
    df['Sentiment_Label'] = [x.get('label') for x in df['Sentiment']]
    df['Sentiment_Score'] = [x.get('score') for x in df['Sentiment']]
    df["Sentiment_Score"] = np.where(df["Sentiment_Label"] == "NEGATIVE", -(df["Sentiment_Score"]), df["Sentiment_Score"])
    return df

def clean_data(text):
    """
    Cleans the texts to the point where it only contains words.

    Parameters
    ----------
    text : String
        Text that you want to clean

    Returns
    -------
    A String with the cleaned text

    """
    text = text.lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text)
    text = re.sub("  ", " ",text)
    stop = stopwords.words('english')
    new_text = " ".join([word for word in text.split() if word not in (stop)])
    return(new_text)

def create_wordcloud(text):
    """
    Creates a wordcloud of the inputed string of cleaned words.

    Parameters
    ----------
    text : String
        The string result from the clean_data function

    Returns
    -------
    A plot of the wordcloud, where the word sizes depend on
    the frequency at which the word occurs.

    """
    word_cloud = WordCloud(
        width=3000,
        height=2000,
        random_state=1,
        background_color="grey",
        colormap="Pastel1",
        collocations=True,
        stopwords=STOPWORDS,
        ).generate(text)

    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()


def Word_Frequency(text, n_words):
    """
    Goes through a string of words and creates
    a list of the 10 most frequent and plots a bar chart
    
    Parameters
    ----------
    text : String
        A string of words, which was cleaned for
        all unecessary symbols and stopwords
    
    n_words : Integer
        How many words you want to include in the final table.
    Returns
    -------
        Creates a bar plot of the 10 most frequent words.
    """
    wordcount = {}
    for word in text.lower().split():
        
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
            # Print the 10 most common word
    n_print = n_words
    word_counter = collections.Counter(wordcount)
    # Create a data frame of the most common words 
    # And draw a bar chart
    lst_all = word_counter.most_common(n_print)
    Word_Count_all = pd.DataFrame(lst_all, columns = ['Word', 'Count'])
    Word_Count_all.plot.bar(x='Word',y='Count')
    
def create_mean(df):
    """
    takes a df and returns the mean value of the 5th column

    Parameters
    ----------
    df : Pandas Data Frame
        here: data_df

    Returns
    -------
    return: float 
    mean value for the analyzed column

    """
    return df.iloc[:,4].mean() 

def sentiment_select(df, sentiment):
    """
    applys the clean_data function onto a df and adds all the clean texts of a given column based on the selected Sentiment
    (input dataframe needs the columns "Sentiment_Lable" and "Clean_Text")
    --> Sentiment_Label determines if the Clean_Text gets used
    --> Clean_Text is the text that gets cleaned and added to the output

    Parameters
    ----------
    df : Pandas Data Frame
        here: data_df
    sentiment : txt 
        (works with either "POSITIVE" or "NEGATIVE" as input)

    Returns
    -------
    TYPE: txt
        returns the joined and cleaned text parts from the selected rows 

    """
    return clean_data(' '.join(df[df["Sentiment_Label"] == sentiment]["Clean_Text"]))

def plot_cloudfrequenzy(df, startdate, stopdate, sentiment, n_words):
    """
    Void Function to print the wordcloud and the n (n_words) most used words of the text

    Parameters
    ----------
    df : Pandas Data Frame
        here: data_relevant
    startdate : txt
        text string of the start date of the analysis period
    stopdate : txt
        text string of the stop date of the analysis period
    sentiment : txt
        (works with either "POSITIVE" or "NEGATIVE" as input)
    n_words : int
        number of words that get shown in the frequency tabel

    Returns
    -------
    None.

    """
    create_wordcloud(' '.join(df[(df["date"] > startdate) & (df["date"] < stopdate)][sentiment]))
    Word_Frequency(' '.join(df[(df["date"] > startdate) & (df["date"] < stopdate)][sentiment]), n_words)


#%% SENTIMENT ANALYSIS execution code:

data_relevant = pd.read_csv('Texts.csv')

# trying to create the df´s inside the df all at once
data_relevant["data_df"] = data_relevant["words"].map(lambda text: cut_sentence(text))

# loading the NLP model
nlp_sentiment = pipeline("sentiment-analysis")

# maps the create_sentiment function on all rows of the data_relevant df
data_relevant["data_df"] = data_relevant["data_df"].map(lambda df: create_sentiment(df))

# Use all the sentiment values of the data_df´s to create a mean value for every article in the data_relevant df
data_relevant["mean_rating"] = data_relevant["data_df"].map(lambda df: create_mean(df))


#Create 2 collumns in the data_relevant df with all the positive and all the negative words of the corresponding data_df´s
data_relevant["POSITIVE"] = data_relevant["data_df"].map(lambda df: sentiment_select(df, "POSITIVE"))
data_relevant["NEGATIVE"] = data_relevant["data_df"].map(lambda df: sentiment_select(df, "NEGATIVE"))

#Creates the wordcloud and the word frequency tabel for the first Period of the analyzed time period (Jan 2010 - April 2014)
plot_cloudfrequenzy(data_relevant, "2010-01-01", "2014-04-19", "NEGATIVE", 20)
plot_cloudfrequenzy(data_relevant, "2010-01-01", "2014-04-19", "POSITIVE", 20)

#Creates the wordcloud and the word frequency tabel for the second Period of the analyzed time period (April 2014 - February 2022)
plot_cloudfrequenzy(data_relevant, "2014-04-20", "2022-02-21", "NEGATIVE", 20)
plot_cloudfrequenzy(data_relevant, "2014-04-20", "2022-02-21", "POSITIVE", 20)

#Creates the wordcloud and the word frequency tabel for the last Period of the analyzed time period (February 2022 - Dezember 2022)
plot_cloudfrequenzy(data_relevant, "2022-02-21", "2022-12-30", "NEGATIVE", 20)
plot_cloudfrequenzy(data_relevant, "2022-02-21", "2022-12-30", "POSITIVE", 20)


#%% Plotting the means over time

# Dates used:
# 27.02.2014 --> Start of full-scale escalation in Donbas
# 24.02.2022 --> Invasion of Ukraine

# create a dataframe with dates as index and means as x values
# also converting the dates into plotable values
df_plot = data_relevant[["date", "mean_rating"]]
df_plot['date'] = df_plot['date'].apply(pd.Timestamp)
df_plot = df_plot.set_index('date')

# creating the plot for the means over time including the 2 Key events:
# 27.02.2014 --> Start of full-scale escalation in Donbas
# 24.02.2022 --> Invasion of Ukraine
df_plot.plot(marker='o', legend=None)
plt.title('Sentiment Means Plotted Over Time')
plt.xlabel('Time')
plt.ylabel("Mean Sentiment")
# add a black dotted line at going through 0
plt.axhline(y=0, ls="--", color="black")
# add a red line for the two key event dates
plt.axvline(pd.Timestamp('2022-02-24'), color='r', label='Invasion of Ukraine')
plt.axvline(pd.Timestamp('2014-02-27'), color='r')
plt.show()
