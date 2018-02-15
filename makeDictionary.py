# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:38:56 2018
@author: murali.sai
"""
# =============================================================================
# Types of dictionaries:
    # FOOD ITEMS for spellCheck(Temporary): paneer, palak, yogurt, rice, biryani,etc
    # SEPERATORS: ki, ke, ka, and, aur, with, ;, etc
    # UNITS of MEASUREMENT: ml, litres, grams, ounces, pounds, etc
    # QUANTITIES: Integers(1,2,3,23,500,etc), Words(five hundred,twenty five, etc)
    # FOOD ITEM DETAILS: Yet to decide on it
# =============================================================================
    
from ScrapeFoodNames import scrapeFoodNames
from nltk.tokenize import word_tokenize #, wordpunct_tokenize
import re, string
import pickle as pk

trueSpells = [];
trueFullNames = []; #Like 'tomato_soup'
def basic_process_text(allW): 
    print("make sure that inut to basic_process_text() is a LIST of text logs");
    all_punctuations = string.punctuation
    #selected_punctuations = re.sub(r'(\&|\-|\.)', "", all_punctuations)
    selected_punctuations = re.sub(r'()', "", all_punctuations)
    punctuation_regex = re.compile('[%s]' % re.escape(selected_punctuations))
    for i, sent in enumerate(allW):
        #Remova all punctuation terms
        allW[i] = punctuation_regex.sub(" ",allW[i])
        #Seperate numerals with alphabets 
        allW[i] = ' '.join( re.findall(r'[A-Za-z]+|\d+', allW[i]));
    return allW;
def makeDictionary():
###################################################################################
#    #Scarping the data
#    fullScrapedData = scrapeFoodNames(fullScrapedData);
#    with open('foods2.txt','rb') as infile:
#        fullScrapedData = pk.load(infile)
#        fullScrapedData = fullScrapedData[:962]
#        re.findall('\((.*?)\)','a(b)(c)')
###################################################################################
#    #Loading a file
#    thefile = open('./data/manjulaKitchen.txt', 'r')
#    temp1 = thefile.read().split("\n")
#    thefile.close()
#    temp2 = [];
#    for item in temp1: #Items that mean the same
#        if '(' in item and ')' in item:
#            outside_par = re.sub(r"\s?\(.*?\)", r"", item).lower().strip()
#            inside_par = re.findall('\((.*?)\)',item) #Can be alist
#            if outside_par not in temp2:
#                temp2.append(outside_par)
#            for e_extr in inside_par:
#                if e_extr.lower().strip() not in temp2:
#                    temp2.append(e_extr.lower().strip())
#        else:
#            if item.lower().strip() not in temp2:
#                temp2.append(item.lower().strip())            
#    temp3 = [];
#    for item in temp2:
#        new_items = item.split(' - '); #Items that mean the same
#        for n_items in new_items:
#            temp3.append(n_items);
#    fullScrapedData = [];
#    for text in temp3:
#        if text.lower().strip() not in fullScrapedData:
#            fullScrapedData.append(text.lower().strip());
###################################################################################
    #Loading a file
    thefile = open('./data/testFoods.txt', 'r')
    fullScrapedData = thefile.read().split("\n")
    thefile.close();
    for ind, text in enumerate(fullScrapedData): 
        fullScrapedData[ind]  = text.replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(',', ' ').lower().strip()
###################################################################################    
    #Preprocessing # Replace all punctuations with 'blank_space':
    allW = basic_process_text(fullScrapedData)    
    wordsSeperated = [];
    wordCount = {}; # Count frequency of each word and delete words with frequency less than 1
    for name in allW:
        tokens = word_tokenize(name);
        for token in tokens:
            if token not in wordsSeperated:     
                wordsSeperated.append(token);
                wordCount[token]=1;
            else:
                wordCount[token]+=1
#    for name in allW:
#        for token in word_tokenize(name):
#            if wordCount[token]<2 or len(token)<3:
#                allW.remove(name);
#                break;
###################################################################################
    # Define a custom dictionary
    tempList = ['the','low','high',
                'lunch','brunch','breakfast','dinner','snacks',
                'eat','ate','had','drink','roast','juice','food',
                'butter','cheese','bread','chai','coffee',
                'milk','skim','fat','raw','buttermilk',
                'idlY','DoSa','Sambhar','vada',
                'apple','mango','watermelon',
                'onion','tomato','potato','peas',
                'Biryani','rice','sprouts','bread','chaI',
                'sabji',
                'ml','litres','slices','cup','bowl',
                'some','little','huge','half','medium','large','high','low',
                'with','in','together','mixed','real','and']
    tempList = [];
    for word in tempList:
        word = word.lower().strip();
        if word not in trueSpells:
            trueSpells.append(word)
    for word in wordsSeperated:
        word = word.lower().strip();
        if word not in trueSpells: #and not word.isdigit():
            trueSpells.append(word)
    #trueSpells = np.concatenate((trueSpells, wordsSeperated))
    ###################################################################################
    # Add '_' this punctuation in between words of allW: "tomato soup"---> "tomato_soup"
    trueFullNames = allW
    #for ind, text in enumerate(trueFullNames):
    #    trueFullNames[ind] = trueFullNames[ind].replace(' ','_')
    #Yet to be modified
    return trueSpells, trueFullNames

