#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat Nov  5 15:16:17 2022
@author: 'Tim Gruber, Domenico Maffei, Mason Tjuanta, Michael Birkenzeller'

"""
#%% All the relevant Libraries

# For the missing libraries use: pip install
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np

#%% Function to crawl for Article Ids

def crawl_article_id(start_page=1, end_page=9):
    """
    This code is specific to the website:
    http://en.kremlin.ru
    and uses the page:
    http://en.kremlin.ru/catalog/countries/UA/events
    --> we use this subpage where the filters are already set to select only articles that fall into the
    thread with the topic "Ukraine". This reduces the amount of data considerably and since we are interested
    in the relationship between Russia and Ukraine and the used wording of Putin in regard to the topics 
    concerning Ukraine this should help us achieve that goal without unneccessary data analysis.

    Returns a list with the article ids from the desired start page till the desired end page in English.

    Parameters
    ----------
    start_page : Integer, optional
        DESCRIPTION. The default is 1.
    end_page : Integer, optional
        DESCRIPTION. The default is 9 
    ("at the time of publishment this default selection contains the disered articles")

    Returns
    -------
    return: list
        A List of Article Ids.

    """

    '''
    Since the parsed webside doesn´t  allow access without providing headers data or using eg. a headless/icognito 
    chrome driver (like provided by selenium) we provide this arificially, and therefore pretending to be a normal 
    web user to circumvent the protection against web scraping
    '''
    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    }
 
    # going from page to page to save the article ids in the data_list
    data_list = []
    for i in range(start_page, end_page+1):
        link = "http://en.kremlin.ru/catalog/countries/UA/events/page/" + str(i)
        response = requests.get(link, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
    
        # adding the content from "href" since the article ids are written there
        for data_id in soup.find_all('a'):
            if data_id.get('href') != None:
                data_list.append(data_id.get('href'))
    
    # since there are a few other data points in "href" we need to delete the unneccesary data to only keep the ids
    data_id_list = []
    for i in range(0, len(data_list)):
        if (data_list[i][0:28] == "/catalog/countries/UA/events") and (data_list[i][-5:].isnumeric()):
            data_id_list.append(data_list[i][-5:])
            
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

    # creating the dataframe to save the parsed information
    df = pd.DataFrame(columns = ["data_id", "date", "text"])
    
    # again providing the headers file
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    }
 
    for i in range(0, len(data_id_list)):   
    
        '''
        unfortunately due to the higher amount of accesses to the id we get again problem with the server, we discussed the usage
        of IP rotation mechanisms but decided on simply implementing a sleep time that also achieves the goal of getting the information
        and even though it slows down the process considerably ... the solution is free of charge - therefore we sleep every 10s id for 15s
        '''
        if i % 10 == 0:
            time.sleep(15)
            
        link = "http://en.kremlin.ru/catalog/countries/UA/events/" + str(data_id_list[i])
        response = requests.get(link, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # downloading the text paragraph by paragraph, considering only parts spoken by putin (we still also endup with the end credits)
        new_text = ""
        '''
        since the articles switch between press statements, speeches by putin and conversations with putin and we decided to analyze
        the words of putin only we only save the text if Putin speaks, the program identifies therefore the current speaker through the 
        bolt printed speaker names provided and only records if Putin is speaking (speaker = True)
        
        '''
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
         
        # since the end credits always start with "Published" we only save the part until that point to end up with the sole speech by putin            
        seperator = "Published"
        
        try:           
           df = pd.DataFrame(np.insert(df.values, len(df), [data_id_list[i], soup.find("time").get('datetime'), new_text.rsplit(seperator, 1)[0]], axis=0))
        except:
            df = pd.DataFrame(np.insert(df.values, len(df), [data_id_list[i], "error", new_text.rsplit(seperator, 1)[0]], axis=0))
           
    df = df.rename(columns={0:"data_id", 1:"date", 2:"words"})
    
    return(df)

#%% Crawling Excution and saving .csv with only relevant Texts

# Get all the ids from the articles pages 1-9==> until 05/2012 maybe increase it a bit?:
article_ids = crawl_article_id(1,9)

# Get all the texts in p from the Wbsite that start with "Putin" in b
# (This only inlcudes the speeches of Putin):
data_df = crawl_article_text(article_ids)

# Removing all empty strings from the dataframe
# (those empty strings stem from press statements not done by putin, and therefore information that is not of interest to us)
data_df.replace("", float("NaN"), inplace=True)
data_relevant = data_df.dropna()

# Create the .csv to access the data for further analysis
data_relevant.to_csv('Texts.csv', index=False)


