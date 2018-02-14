# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:11:30 2018
@author: Lenovo
"""
import numpy as np
from makeDictionary import makeDictionary
from nltk.tokenize import wordpunct_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
#sorted({'a': 1, 'c': 3, 'b': 2}, reverse=True)

# Cosine Similarity and Top_k similar foods
def similarFoods(givenFood, foodDict, top_k):
    # Approach:
    # Make 2 sets:
        # One Set consisting of all food Items that have at least one of the tokens in givenFood
        # Another: All other food Items in foodDict
    #givenFood = r'bread burfi'
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
def similarFoodsHistory(givenFoods, foodDict, top_k, method=1):
    #givenFoods = [r'masala vada', r'aloo gobi', r'bread burfi', r'help']
    #Three ways:
            #Take avg of all givenFoods' vecs and get simFoods to that vec
            #Get SimFoods for each item in givenFoods seperately and randomly pick 3 from each set...
                #...and randomly coose 10 from resulting list
            #Consider only some of all givenFoods and then take avg and proceed
    if method==1:
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
        simScore = {};
        for key in (foodDict.keys()):
            simScore[key] = cosine_similarity(foodDict[key], thisVec)[0][0]
        simFoods = sorted(simScore, key=simScore.get, reverse=True)[:top_k+1]
        simFoods = simFoods[1:]  
        return simFoods
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
    del tf_mat, tf_pca, individualNames
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
    givenFood = r'coconut chutney';
    givenFood = r'apple vegan cake';
    #givenFood = r'bread burfi';
    simFoods1 = similarFoods(givenFood, foodDict, 20)
    
    givenWord = r'eggless'
    simWords1 = similarWords(givenWord, wordDict, 20)
    
    givenFoods = [r'masala vada', r'aloo gobi', r'bread burfi', r'help'];
    simFoods2 = similarFoodsHistory(givenFoods, foodDict, 20, method=1)


