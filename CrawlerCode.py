#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Nov  5 15:16:17 2022
@author: Michael Birkenzeller

Not recommended to run this code, as the russian website doesn't like it, 
this file is only for documentation purpose.

"""
#%% All the relevant Libraries

# For the missing libraries use: pip install
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import datetime as dt
import pandas as pd
import time
import numpy as np

#%% Function to crawl for Article Ids
def crawl_article_id(start_page=1, end_page=4):
    """
    This code is specific to the website:
    http://en.kremlin.ru
    and uses the page:
    http://en.kremlin.ru/catalog/countries/UA/events

    Returns a list with the article ids from the desired start page till
    the desired end page in English.

    Parameters
    ----------
    start_page : Integer, optional
        DESCRIPTION. The default is 1.
    end_page : Integer, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    A List of Article Ids.

    """
    begintime = dt.datetime.now()
    # Headless/incognito Chrome driver
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument('headless')
    driver = webdriver.Chrome(executable_path='CHROMEDRIVER_PATH',
                              chrome_options=chrome_options)
    
    data_list = []
    for i in range(start_page, end_page+1):
        link = "http://en.kremlin.ru/catalog/countries/UA/events/page/" + str(i)
        driver.get(link)
        # time.sleep(2)
    
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
    
        for data_id in soup.find_all('a'):
            if data_id.get('href') != None:
                data_list.append(data_id.get('href'))
    
    data_id_list = []
    for i in range(0, len(data_list)):
        if (data_list[i][0:28] == "/catalog/countries/UA/events") and (data_list[i][-5:].isnumeric()):
            data_id_list.append(data_list[i][-5:])
            
    
    print("Runtime crawl_article_id for " + str(end_page+1 - start_page) + " pages: " + str(dt.datetime.now() - begintime))
    
    return(data_id_list)

# Function to crawl for the texts given the article ids only including Putins speeches
#%% Function to crawl for Article Texts and Dates

def crawl_article_text(data_id_list): 
    """
    This code uses all the article Ids to get all the respective texts in:
    http://en.kremlin.ru/catalog/countries/UA/events
    
    I then creates a Dataframe with the columns: article id, dates, words.
    Parameters
    ----------
    data_id_list : list
        A list of all the article ids for which to crawl for.

    Returns
    -------
    Dataframe with the columns: article id, dates, words.

    """
    
    begintime = dt.datetime.now()

    df = pd.DataFrame(columns = ["data_id", "date", "text", "word_list", "sentiment"])
    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    }
 
    for i in range(0, len(data_id_list)):   #len(data_id_list)
    
        if i % 10 == 0:
            time.sleep(15)
            
        link = "http://en.kremlin.ru/catalog/countries/UA/events/" + str(data_id_list[i])
        response = requests.get(link, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        #downloading the text paragraph by paragraph, considering only parts spoken by putin (still contains the end part, but 95% there)
        new_text = ""
        speaker = False
        for line in soup.find_all("p"):
            try:
                if "Putin" in line.b.get_text():
                    speaker = True
                else:
                    speaker = False
                if speaker == True:
                    new_text += line.get_text()[len(line.b.get_text())+1:] + "\n"
            except:
                if speaker == True:
                    new_text += line.get_text() + "\n"

        try:           
           df = pd.DataFrame(np.insert(df.values, len(df), [data_id_list[i], soup.find("time").get('datetime'), new_text, False, False], axis=0))
        except:
            df = pd.DataFrame(np.insert(df.values, len(df), [data_id_list[i], "error", new_text, False, False], axis=0))
                
    df = df.rename(columns={0:"data_id", 1:"date", 2:"words", 3:"word_list", 4:"sentiment"})
    
    print("Runtime crawl_article_text for " + str(len(data_id_list)) + " articles: ")
    print(dt.datetime.now() - begintime)
    
    return(df)

#%% Crawling Excution and saving .csv with only relevant Texts

# Get all the ids from the articles pages 1-9==> until 05/2012 maybe increase it a bit?:
article_ids = crawl_article_id(1,9)

# Get all the texts in p from the Wbsite that start with "Putin" in b
# (This only inlcudes the speeches of Putin):
data_df = crawl_article_text(article_ids)

# Removing all empty strings from the dataframe
nan_value = float("NaN")
data_df.replace("", nan_value, inplace=True)
data_relevant = data_df.dropna()

# Create the .csv file for the further analysis
"""
This part is only temporary
Removes the columns word list& sentiment <== these shouldn't be needed anymore right?'
data_relevant = data_relevant.iloc[:, [0, 1, 2]]
"""
data_relevant.to_csv('Texts.csv')
