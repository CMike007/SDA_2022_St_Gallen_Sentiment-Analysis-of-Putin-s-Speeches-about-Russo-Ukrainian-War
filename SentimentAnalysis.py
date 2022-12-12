# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:46:21 2022

@author: timgr

Source of original code: https://github.com/chuachinhon/practical_nlp/blob/master/notebooks/1.0_speech_sentiment_cch.ipynb

Currently it only looks only at the first text 
==> Need to adapt the code so that it does it for all 30 texts
==> Important to allocate each analysis to the specific date of the texts
    to enable an Analysis over time
==> Maybe it's possible to attach all figures in a single line w.r.t. dates'
    this would vsualize all sentiments for each sentences accross time

"""
#%% Import all the libraries

# For the missing libraries use: pip install
from __future__ import print_function
import ipywidgets as widgets
import matplotlib as mpl
import pandas as pd
import plotly
# pip install plotly_express
import plotly_express as px
import plotly.graph_objects as go
import numpy as np
import re

from transformers import pipeline

"""
not sure what the following 3 lines do: (does anyone know? its from the original code)
"""
mpl.rcParams["figure.dpi"] = 300
%matplotlib inline
%config InlineBackend.figure_format ='retina'


#%% code to open the excel file as a pd.dataframe 
# (make sure wd is set & that csv file is saved there)
# also remove 1st row as it is the same as the index
data_relevant = pd.read_csv('Texts.csv')
data_relevant = data_relevant.iloc[: , 1:]

# Taking only the texts as a series
texts = data_relevant["words"]


#%% Data Preparation for the Political Sentiment Analysis

# Creates a Dataframe with 1 column containing all the sentences in the text
"""
Original code splitted for paragraphs using "/n/n"
this didn't work with our texts so I changed the split to "."
==> This means that paragraphs are in our case sentences

==> should also be made into a function
"""
pm = (
    pd.DataFrame(data_relevant["words"][0:1].str.split(".").tolist(), index=data_relevant[0:1].index)
    .stack()
    .reset_index()
    .rename(columns={0: "Paras"})
    .drop("level_0", axis=1)
    .drop("level_1", axis=1)
)

# Simple cleaning of text ahead of sentiment analysis
"""
works well but has some double spaces between words <-- Note in case it becomes an issue
"""
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = text.strip(" ")
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Adds a second column to the Dataframe with the cleaned sentences
pm["Clean_Text"] = pm['Paras'].map(lambda text: clean_text(text))

# Creates a list containing all the cleaned sentences (needed as input for the sentiment analysis)
corpus = list(pm['Clean_Text'].values)


#%% The Political Sentiment Analysis
"""
Quote from study:
"You just need 3 lines of code to execute the sentiment analysis task via HF's pipeline, 
which does all the heavy lifting for the complex code. I did not fine tune the model 
for Singapore politics or Covid-19."

==> should we finetune it or not? I suggest not but it depends on the results
==> If not we get this feedback in the console:
    "No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended."
    
"""

# Creates the sentiments for all sentences seperatly (currently default model <-- change?)
nlp_sentiment = pipeline(
    "sentiment-analysis"
)

# Attach the results to the Dataframe
pm["Sentiment"] = nlp_sentiment(corpus)

# The pipeline's sentiment analysis output consists of a label and a score
# Extract them into separate columns in the Dataframe
pm['Sentiment_Label'] = [x.get('label') for x in pm['Sentiment']]
pm['Sentiment_Score'] = [x.get('score') for x in pm['Sentiment']]

# Creating a sentiment analysis result table including only the Clean Sentences, sentiment labels and scores
# Works well but there are still some \n(i think these create some double spaces, not sure, bt they don't really bother) --> see data cleaning
cols = ["Clean_Text", "Sentiment_Label", "Sentiment_Score"]

df = pm[cols].copy()

# Changing all the scores of value == NEGATIVE from + to -
df["Sentiment_Score"] = np.where(
    df["Sentiment_Label"] == "NEGATIVE", -(df["Sentiment_Score"]), df["Sentiment_Score"]
)
# Optional step to save results as a csv
# df.to_csv('pm_sentiment.csv', index=False)


#%% Visualization with Plotly's Heatmap

# Creating a figure with the results using Plotly's Heatmap

fig = go.Figure(
    data=go.Heatmap(
        z=df["Sentiment_Score"],
        x=df.index,
        y=df["Sentiment_Label"],
        colorscale=px.colors.sequential.RdBu,
    )
)

fig.update_layout(
    title=go.layout.Title(
        text="1st text test"
    ),
    autosize=False,
    width=1200,
    height=600,
)

fig.update_layout(yaxis_autorange = "reversed")

fig.show()

# Run this line to save figure as an html file
# fig.write_html("pm.html")


#%% Summary Table

def Summary_Table(df):
    # Value Count, Mean & std for all sentences
    Count_All = len(df)
    Percent_All = 1
    Mean_All = df["Sentiment_Score"].mean()
    std_All = df["Sentiment_Score"].std()

    # Value Count & Percentage, Mean & std for negative sentences 
    # mean and std is a bit useless here I think
    Count_Neg = df.loc[df['Sentiment_Label'] == 'NEGATIVE', 'Sentiment_Label'].value_counts()[0]
    Percent_Neg = round(df.loc[df['Sentiment_Label'] == 'NEGATIVE', 'Sentiment_Label'].value_counts()/len(df), 2)[0]
    Mean_Neg = round(df.loc[df['Sentiment_Label'] == 'NEGATIVE', 'Sentiment_Score'].mean(), 2)
    Std_Neg = df.loc[df['Sentiment_Label'] == 'NEGATIVE', 'Sentiment_Score'].std()


    # Value Count & Percentage, Mean & std for positive sentences
    # mean and std is a bit useless here I think
    Count_Pos = df.loc[df['Sentiment_Label'] == 'POSITIVE', 'Sentiment_Label'].value_counts()[0]
    Percent_Pos = round(df.loc[df['Sentiment_Label'] == 'POSITIVE', 'Sentiment_Label'].value_counts()/len(df), 2)[0]
    Mean_Pos = round(df.loc[df['Sentiment_Label'] == 'POSITIVE', 'Sentiment_Score'].mean(), 2)
    Std_Pos = df.loc[df['Sentiment_Label'] == 'POSITIVE', 'Sentiment_Score'].std()

    # Summary Table

    table = pd.DataFrame({"Overview": (Count_All, Percent_All, Mean_All, std_All),
                          "Negative": (Count_Neg, Percent_Neg, Mean_Neg, Std_Neg),
                          "Positive": (Count_Pos, Percent_Pos, Mean_Pos, Std_Pos)
                              })
    # Create the Index
    index = pd.Index(["Count", "Percentage", "Mean", "Standard Deviation"])
    table = table.set_index(index)
    return round(table, 2)

Summary = Summary_Table(df)

