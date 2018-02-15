# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:11:30 2018
@author: Lenovo
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from makeDictionary import makeDictionary, basic_process_text
from spellCorrection import DL_distanceBased

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
#sorted({'a': 1, 'c': 3, 'b': 2}, reverse=True)

# Cosine Similarity and Top_k similar foods
def similarFoods(givenFood, foodDict, top_k):
    try:
        thisVec = foodDict[givenFood];
        simScore = {};
        for key in (foodDict.keys()):
            simScore[key] = cosine_similarity(foodDict[key], thisVec)[0][0]
        del thisVec, key
    except Exception as e:
        print(e)    
    simFoods = sorted(simScore, key=simScore.get, reverse=True)[:top_k+1]
    simFoods = simFoods[1:]  
    return simFoods
def similarWords(givenWord, wordDict, top_k):
    try:
        thisVec = wordDict[givenWord];
        simScore = {};
        for key in (wordDict.keys()):
            simScore[key] = cosine_similarity(wordDict[key], thisVec)[0][0]
        del thisVec, key
    except Exception as e:
        print(e)    
    simWords = sorted(simScore, key=simScore.get, reverse=True)[:top_k+1]
    simWords = simWords[1:]  
    return simWords
def similarFoodsHistory(givenFoods, foodDict, inputString, allWords, top_k, method=1):
    # APPROACH
    # Considering all tokens in inputStringfollowing set of foodItems is made.
    # Set of all items in foodDict such that each item has one or more tokens of inputString in it.
    # Then the top_k foodItems that are closest to the pastVector of user are taken
    # If #foodItems in that list are less that top_k, top matches from other food item are added to the list.

    #WAYS TO UTILIZE USER HISTORY
    # Take avg of all givenFoods' vecs and get simFoods to that vec
    # Get SimFoods for each item in givenFoods seperately and randomly pick 3 from each set and randomly choose 10 from resulting list
    # Consider only some of all givenFoods and then take avg and proceed

    # ALGORITHM
    if method==1:
        #Spell Correction
        textLogs = [inputString]
        textLogs = basic_process_text(textLogs);
        for ind, log in enumerate(textLogs):
            #Assumption from hereon is that the logs can ONLY contain lowercase_alphabets, '_', and 'numerals'
            dl = DL_distanceBased(allWords, log); 
            textLogs[ind] = dl.spellCorrect();
            #print('Cost to change: ', sum(dl.minScores)); #print('Spell Correction Done'); print('Before: ', log); print('After: ',textLogs[ind]);       
        inputString = textLogs[0];
        #Computing userVector from past food items he/she ate
        thisVec, histCnt = [], 0;
        for hist in givenFoods:
            if thisVec==[]:
                try:
                    thisVec = foodDict[hist]; histCnt+=1;
                except Exception as e:
                    print(e); continue;
            else:
                try:
                    thisVec += foodDict[hist]; histCnt+=1;
                except Exception as e:
                    print(e); continue;  
        if histCnt==0:
            raise Exception('Atleast one of the input foods should be in the dictionary!!')
        thisVec/=histCnt; #Taking average of past food items
        #Set 1 as described above
        relevant_foodDict = {};
        inputString_tokens = wordpunct_tokenize(inputString)
        for key in foodDict.keys():
            key_tokens = wordpunct_tokenize(key);
            unique_tokens = [];
            sim_word_count = 0;
            for token in inputString_tokens:
                if token not in unique_tokens and token in inputString_tokens and token in key_tokens:
                    unique_tokens.append(token);
                    sim_word_count += np.min( [unique_tokens.count(token), inputString_tokens.count(token)] );
            if sim_word_count>=1:
                relevant_foodDict[key] = sim_word_count;
        #Finding Similarity index for relevant_foodDict #Currently, no use with 'sim_word_count'
        simFoods = [];
        relevant_foods_num = top_k; others_foods_num = [];
        relevant_simScore = {}; 
        for key in (relevant_foodDict.keys()):
            relevant_simScore[key] = cosine_similarity(foodDict[key], thisVec)[0][0]
        relevant_foods_num = np.min( [relevant_foods_num, len(relevant_simScore.keys())] );
        others_foods_num = top_k - relevant_foods_num;
        simFoods = sorted(relevant_simScore, key=relevant_simScore.get, reverse=True)[:relevant_foods_num]       
        others_simScore = {};
        for key in foodDict.keys():
            if key not in relevant_foodDict.keys():
                others_simScore[key] = cosine_similarity(foodDict[key], thisVec)[0][0]
        simFoods+=sorted(others_simScore, key=others_simScore.get, reverse=True)[:others_foods_num]                    
        return simFoods;
    elif method==2:
        print('Under Construction');
    elif method==3:
        print('Under Construction');
    
if __name__=='__main__':    
    # PCA on the ManjulaKitchen's data
    individualNames, fullNames = makeDictionary();
    #Generate foodDict
    wordIndices = {};
    for ind, word in enumerate(individualNames):
        wordIndices[word] = ind;
    tf_mat = np.zeros((len(fullNames),len(individualNames)));
    for ind, text in enumerate(fullNames):
        tokens = wordpunct_tokenize(text)
        for token in tokens:
            tf_mat[ind][wordIndices[token]]+=1
    del ind, text, tokens, token, word
    tf_pca = PCA(n_components = 10).fit_transform(tf_mat)
    for row_ind in range(np.shape(tf_pca)[0]):
        tf_pca[row_ind]/=np.linalg.norm(tf_pca[row_ind,:])
    del row_ind;
    foodDict = {};
    for ind, name in enumerate(fullNames):
        foodDict[name] = tf_pca[ind,:];
    del ind, name
    del tf_mat, tf_pca
    #Generate wordDict
    wordDict_temp = np.zeros( ( len(wordIndices.keys()),len(wordIndices.keys()) ) )
    for ind, text in enumerate(fullNames):
        tokens = wordpunct_tokenize(text)
        n_tokens = len(tokens)
        for i in np.arange(0, n_tokens-1, 1):
            for j in np.arange(i+1, n_tokens, 1):
                wordDict_temp[wordIndices[tokens[i]], wordIndices[tokens[j]]] += 1;
                wordDict_temp[wordIndices[tokens[j]], wordIndices[tokens[i]]] += 1;
    del ind, text, tokens, n_tokens, i, j
    wordDict_temp = PCA(n_components = 10).fit_transform(wordDict_temp);
    for row_ind in range(np.shape(wordDict_temp)[0]):
        wordDict_temp[row_ind]/=np.linalg.norm(wordDict_temp[row_ind,:]) 
    del row_ind;
    wordDict = {};
    for key in wordIndices.keys():
        ind = wordIndices[key];
        wordDict[key] = wordDict_temp[ind,:]
    del wordDict_temp, ind, key
        
    #Checks
#    givenFood = r'coconut chutney'; #givenFood = r'apple vegan cake'; #givenFood = r'bread burfi';
#    simFoods = similarFoods(givenFood, foodDict, 20)
#    givenWord = r'eggless'
#    simWords1 = similarWords(givenWord, wordDict, 20)
    
    #testFoods Example 1 
    givenFoods = [r'aloo paratha', r'2 egg white and onion omelette', r'moong dal fry']
    inputString = r'parata'
    simFoods1 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)

    
    #givenFoods = [r'almond cashew burfi', r'masala vada', r'almond chikki',r'aloo gobi', r'bread burfi', r'help'];
    #Example 1
    givenFoods = [r'aloo paratha', r'bread pakoras']
    inputString = r'almonds'
    simFoods1 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)
    #Example 2
    givenFoods = [r'eggless omelet',r'eggless pancake',r'bread paneer rolls']
    inputString = r'paaneeer'
    simFoods2 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)
    #Example 3
    givenFoods = [r'peanut chutney']
    inputString = r'yogort'
    simFoods3 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)
    #Example 4
    givenFoods = [r'apple coconut barfi', r'coconut almond burfi', r'punjabi flatbread']
    inputString = r'chutiney'
    simFoods4 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)     
    #Example 5
    givenFoods = [r'vegetable curry', r'sweet boondi', r'sweet flatbread']
    inputString = r'partha'
    simFoods5 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)     
    #Example 6
    givenFoods = [r'aloo naan', r'apple bread rolls', r'yogurt sandwich', r'shahi paneer']
    inputString = r'parotha'
    simFoods6 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1)   
    #Example 7
    givenFoods = [r'apple coconut barfi', r'apple bread rolls', r'yogurt sandwich', r'apple vegan cake']
    inputString = r'parotha'
    simFoods7 = similarFoodsHistory(givenFoods, foodDict, inputString, individualNames, 10, method=1) 
  


