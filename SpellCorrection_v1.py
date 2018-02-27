# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:28:48 2018
@author: murali.sai
Sample Run:
    dl = DL_distanceBased(dictionary);
    dists = dl.cost_between_chars;
    dl.spellCorrect(log);
"""
# Libaries
import numpy as np
import string
import re
from nltk.tokenize import wordpunct_tokenize
class DL_distanceBased(object):
    #Assumptions:Input can only have ALPHABETS_lowercae, numbers(0-9), and punctuations('_')
    def __init__(self, keyboardType='QWERTY'): #a dict of words
        #Inputs
        self.keyboardType = keyboardType;
        # Constants
        self.DP_STATUS_CHECK, self.MAX_TOKEN_LENGTH = -100, 75
        self.TOTAL_CHAR_TYPES = 256
        self.keypad_alpha = [['q','w','e','r','t','y','u','i','o','p'],
                             ['a','s','d','f','g','h','j','k','l'],
                             ['z','x','c','v','b','n','m']];
        self.keypad_numerals = ['1','2','3','4','5','6','7','8','9','0'];
        self.keypad_punctuations = string.punctuation;
        #Others
        self.correctedText = [];
        self.minScores = [];
        self.dpMat = np.zeros((self.MAX_TOKEN_LENGTH,self.MAX_TOKEN_LENGTH),dtype='int')
        self.cost_between_chars, self.min_cost_val, self.max_cost_val = [], 1, [];
        #Execute Methods
        self.keyboard_dists();
    def keyboard_dists(self):
        if self.keyboardType!='QWERTY':
            raise Exception('Only *QWERTY* Keyboard type is valid')
        #############################################################
        possible_chars = [];
        for row in self.keypad_alpha:
            for alpha in row:
                possible_chars.append(alpha)
        for numeral in self.keypad_numerals:
            possible_chars.append(numeral);
        for punt in self.keypad_punctuations:
            possible_chars.append(punt);
        #############################################################
        #Distance between {Characters} ONLY
        dist_hor, dist_ver = 2, 2 #Please be careful when changing these
        coords = {};
        for ind, row in enumerate(self.keypad_alpha):
            if ind==0:
                curr_hor, curr_ver = 0, 0;
            elif ind==1:
                curr_hor, curr_ver = dist_hor/2, dist_ver;
            elif ind==2:
                curr_hor, curr_ver = dist_hor+dist_hor/2, dist_ver+dist_ver;
            for alpha in row:
                coords[alpha] = [curr_hor, curr_ver];
                curr_hor+=dist_hor;
        self.cost_between_chars = np.zeros((self.TOTAL_CHAR_TYPES,self.TOTAL_CHAR_TYPES))
        for alpha in coords.keys():
            mn = np.inf
            for others in coords.keys():
                temp_coord_dist = np.array([coords[alpha][0]-coords[others][0],coords[alpha][0]-coords[others][0]])
                score = np.sqrt(temp_coord_dist.dot(temp_coord_dist))
                self.cost_between_chars[ord(alpha)][ord(others)]= score
                if score!=0:
                    mn = np.min([mn,score])
            for others in coords.keys():
                self.cost_between_chars[ord(alpha)][ord(others)]/=mn;
        self.max_cost_val = np.max(self.cost_between_chars)
        #############################################################
        #Distance between {Numerals} ONLY
        for n1 in self.keypad_numerals:
            for n2 in self.keypad_numerals:
                self.cost_between_chars[ord(n1)][ord(n2)] = self.min_cost_val;
        #############################################################
        #Distance between {Punctuations} ONLY
        for n1 in self.keypad_punctuations:
            for n2 in self.keypad_punctuations:
                self.cost_between_chars[ord(n1)][ord(n2)] = self.min_cost_val;
        ############################################################# 
        #Distance between {alphabets} and {Numerals,Punctuations} ONLY
        for n1 in coords.keys():
            for n2 in self.keypad_numerals:
                self.cost_between_chars[ord(n1)][ord(n2)] = self.max_cost_val;
        for n1 in coords.keys():
            for n2 in self.keypad_punctuations:
                self.cost_between_chars[ord(n1)][ord(n2)] = self.max_cost_val;          
        ############################################################# 
        #Distance between {Numerals} and {Punctuations} ONLY
        for n1 in self.keypad_punctuations:
            for n2 in self.keypad_numerals:
                self.cost_between_chars[ord(n1)][ord(n2)] = self.max_cost_val;
        #############################################################  
        self.cost_between_chars = self.cost_between_chars+self.cost_between_chars.T;          
        print('self.cost_between_chars COMPUTED');
        return 
    def recur(self, strTrue, strError, indTrue, indError):
        # A user is not expected to make a mistake by omitting the first few words
        # case when strTrue:'home', strError:'ome'
        if indError==-1 and indTrue>=0: #Implies adding of some characters in strError's beginning which is less probable
            return self.max_cost_val;
        elif indError==-1 and indTrue==-1: # think of 'ome' and 'omelette'
            return 0;
        elif indError>=0 and indTrue==-1: #Implies deletion of some characters in strError's beginning which is less probable
            return self.max_cost_val;
        elif self.dpMat[indTrue][indError]!=self.DP_STATUS_CHECK:
            return self.dpMat[indTrue][indError]
        elif strTrue[indTrue]==strError[indError]: #Same characters
            self.dpMat[indTrue][indError] = 0 + self.recur(strTrue, strError, indTrue-1, indError-1);
            return self.dpMat[indTrue][indError];
        else: #Cases of transposition or Replace or Delete or Insert 
            if indTrue and indError and strTrue[indTrue-1]==strError[indError] and strTrue[indTrue]==strError[indError-1]:
                self.dpMat[indTrue][indError] = np.min([ 
                    self.min_cost_val + self.recur(strTrue, strError, indTrue-2, indError-2),    
                    self.cost_between_chars[ord(strError[indError])][ord(strTrue[indTrue])] + self.recur(strTrue, strError, indTrue-1, indError-1),
                    self.min_cost_val + self.recur(strTrue, strError, indTrue, indError-1),
                    self.min_cost_val + self.recur(strTrue, strError, indTrue-1, indError)])
                return self.dpMat[indTrue][indError];  
            else: #Cases of only Replace or Delete or Insert
                self.dpMat[indTrue][indError] = np.min([ 
                    self.cost_between_chars[ord(strError[indError])][ord(strTrue[indTrue])] + self.recur(strTrue, strError, indTrue-1, indError-1),
                    self.min_cost_val + self.recur(strTrue, strError, indTrue, indError-1),
                    self.min_cost_val + self.recur(strTrue, strError, indTrue-1, indError)])
                return self.dpMat[indTrue][indError];
    def compute(self, strTrue, strError):
          lenTrue, lenError = len(strTrue), len(strError)
          if lenTrue>self.MAX_TOKEN_LENGTH or lenError>self.MAX_TOKEN_LENGTH:
              raise Exception('MAX LENGTH OF A TOKEN VIOALTED. Increment self.MAX_TOKEN_LENGTH')
          for i in range(lenTrue):
              for j in range(lenError):
                  self.dpMat[i][j]=self.DP_STATUS_CHECK;
          return self.recur(strTrue, strError, lenTrue-1, lenError-1)
    def all_punctuations(self, token):
        for item in token:
            if item not in string.punctuation:
                return False
        return True
    def spellCorrect(self, sampleText, dictionary):
        self.sampleText = sampleText;
        self.dictionary = dictionary;
        # self.sampleText = ' '.join( re.findall(r'[A-Za-z_]+|\d+', self.sampleText.lower()) ) #Seperate alpah and numerals
        inputTokens = wordpunct_tokenize(self.sampleText.lower());
        outputTokens = []
        # For each token in sampleText, evaluate dist with all items in dictionary if required
        for indT, token in enumerate(inputTokens): #A know token: No modifications required
            if token in self.dictionary or self.all_punctuations(token) or token.isdigit():
                outputTokens.append(token); #print(token)
                self.minScores.append(0);
            #An unkown token
            #Two possibilities: New food item or noisy token
            else: #Find EDIT distance with all know trueSpells and find the word with least distance
                self.itemDists = np.full(len(self.dictionary), np.inf); #Initialization
                for indS, item in enumerate(self.dictionary):
                    strTrue = item.lower().strip();
                    strError = token.lower().strip();
                    dist = self.compute(strTrue, strError)
                    if dist==np.max([len(item), len(token)]): #Corner case when one token is NULL
                        dist = np.inf
                    self.itemDists[indS] = dist
                self.minScores.append(np.min(self.itemDists));
                #outputTokens.append(self.dictionary[np.argmin(self.itemDists)]);
                #If multiple min exists? Select the word with maximum character similarity
                arr = np.where(self.itemDists==np.min(self.itemDists))[0]
                if len(arr)==1:
                    outputTokens.append(self.dictionary[arr[0]])
                else:
                    simi = -1*np.inf
                    for arr_item in arr:
                        string_temp, uniques, simCount = self.dictionary[arr_item], [], 0;
                        for char in string_temp:
                            if char not in uniques:
                                uniques.append(char);
                                simCount+=np.min([len(re.findall(char, string_temp)), len(re.findall(char, token))]);
                        if simCount>simi:
                            appendThis, simi = string_temp, simCount
                    outputTokens.append(appendThis);
            #else:
            #this food item not in out dictionary: Please add it
            #Put it as a new food item if dist with some % of dict elements is greater than some threshold
        self.correctedText = ' '.join(outputTokens)
        print(self.sampleText); print(self.correctedText);
        return self.correctedText