# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:12:45 2018
@author: murali.sai

@Sample Script
newObj = addFoodItems(r'./data/testFoods1.txt', '');
foodIndices = newObj.foodIndices;
foodDict = newObj.foodDict;
wordIndices = newObj.wordIndices;
wordDict = newObj.wordDict;

addUser('murali@12345', 'userIDs.pickle')
deleteUser('murali@12345', 'userIDs.pickle')
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np, re, string, pickle as pk, os
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

#Custom Functions
from SpellCorrection_v1 import DL_distanceBased

#Custom helper functions
def basic_process_text(allW): 
    if not isinstance(allW, list): #True if allW is just 1 string
        raise Exception("make sure that input to basic_process_text() is a LIST of text logs");
    all_punctuations = string.punctuation    #selected_punctuations = re.sub(r'(\&|\-|\.)', "", all_punctuations)
    selected_punctuations = re.sub(r'()', "", all_punctuations)
    punctuation_regex = re.compile('[%s]' % re.escape(selected_punctuations))
    for i, sent in enumerate(allW):
        #Remove all punctuation terms
        allW[i] = punctuation_regex.sub(" ",allW[i])
        #Seperate numerals with alphabets 
        allW[i] = ' '.join( re.findall(r'[A-Za-z]+|\d+', allW[i]));
    return allW;

#Food Item Functionalities
class addFoodItems(object):
    def __init__(self, sourceURL, destinationFolder, pca_dim=10):
        self.pca_dim = pca_dim;
        self.foodDict, self.wordDict = {}, {};
        self.wordIndices, self.foodIndices = {}, {};
        #self.users, self.users_logCount = {}, {}; #User history vector, Count of no of foods logged by user
        #self.sCorrection = []; #Object for class SpellCorrection
        self.readExistingData(destinationFolder);
        self.recomputeVectors(sourceURL);
        self.saveFoodVectors(destinationFolder);
        return
    def loadDataFromFile(self, sourceURL): #r'./data/testFoods1.txt'
        trueWordSpellings = []; foodItemsList = [];
        try: #Loading a file
            thefile = open(sourceURL, 'r')
            fullScrapedData = thefile.read().split("\n")
            thefile.close();
        except Exception as e:
            print('Exception in make Dictionary', e); return;
        #Preprocessing
        for ind, text in enumerate(fullScrapedData): 
            fullScrapedData[ind]  = text.replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(',', ' ').lower().strip()
        allW = basic_process_text(fullScrapedData) 
        # Count frequency of each word
        wordsSeperated = [];
        wordCount = {}; 
        for name in allW:
            tokens = word_tokenize(name);
            for token in tokens:
                if token not in wordsSeperated:     
                    wordsSeperated.append(token); wordCount[token]=1;
                else:
                    wordCount[token]+=1
        # wordDict
        for word in wordsSeperated:
            word = word.lower().strip();
            if word not in trueWordSpellings: #and not word.isdigit():
                trueWordSpellings.append(word)
        # foodDict 
        foodItemsList = [];
        for foodItem in allW:
            if foodItem not in foodItemsList:
                foodItemsList.append(foodItem)        
        return trueWordSpellings, foodItemsList
    def readExistingData(self, destinationFolder):
        try:
            path = os.path.join(destinationFolder, 'foodIndices.pickle')
            with open(path, 'rb') as openedFile:
                self.foodIndices = pk.load(openedFile);openedFile.close();
            print('foodIndices.pickle file existing in destination folder. Loading it.')
        except: 
            self.foodIndices= {};
            print('foodIndices.pickle file not existing in destination folder!')
        try:
            path = os.path.join(destinationFolder, 'wordIndices.pickle')
            with open(path, 'rb') as openedFile:
                self.wordIndices = pk.dump(openedFile);openedFile.close();  
            print('wordIndices.pickle file existing in destination folder. Loading it.')
        except:
            self.wordIndices= {};
            print('wordIndices.pickle file not existing in destination folder!')            
        return
    def recomputeVectors(self, foodsURL):
        idf = {};
        tf_mat, tf_pca = [], [];
        self.wordIndices, self.foodIndices = {}, {};
        self.foodDict, self.wordDict = {}, {};
        #Fetch Data
        trueWordSpellings, foodItemsList = self.loadDataFromFile(foodsURL);
        tf_mat = np.zeros((len(foodItemsList),len(trueWordSpellings)))
        #Index Them
        word_count = len(self.wordIndices.keys());
        for word in trueWordSpellings:
            if word not in self.wordIndices.keys():
                self.wordIndices[word] = word_count; word_count+=1;
        foods_count = len(self.foodIndices.keys());
        for food in foodItemsList:
            if food not in self.foodIndices.keys():
                self.foodIndices[food] = foods_count; foods_count+=1;
        #Find Inverse Document Frequency
        for word in trueWordSpellings:
            idf[word] = 0;
        for name in foodItemsList:
            unique_words = [];
            tokens = wordpunct_tokenize(name);
            for token in tokens:
                if token not in unique_words:
                    unique_words.append(token); idf[token]+=1;
        for key in idf.keys():
            if idf[key]!=0:
                idf[key] = np.log(len(foodItemsList)/idf[key]);
        #Compute TF-IDF and do PCA to find normalized foodVectors
        for foodItem in foodItemsList:
            tokens = wordpunct_tokenize(foodItem)
            for token in tokens:
                tf_mat[self.foodIndices[foodItem]][self.wordIndices[token]]+=(1*idf[token]);
        tf_pca = PCA(n_components = self.pca_dim).fit_transform(tf_mat)
        for row_ind in range(np.shape(tf_pca)[0]):
            tf_pca[row_ind]/=np.linalg.norm(tf_pca[row_ind,:])
        #Save as foodDict
        for foodItem in foodItemsList:
            self.foodDict[foodItem] = tf_pca[self.foodIndices[foodItem],:];
        #Generate wordDict
        wordDict_temp = np.zeros( ( len(self.wordIndices.keys()),len(self.wordIndices.keys()) ) )
        for ind, text in enumerate(foodItemsList):
            tokens = wordpunct_tokenize(text)
            n_tokens = len(tokens)
            for i in np.arange(0, n_tokens, 1):
                for j in np.arange(i+1, n_tokens, 1):
                    wordDict_temp[self.wordIndices[tokens[i]], self.wordIndices[tokens[j]]] += 1;
                    wordDict_temp[self.wordIndices[tokens[j]], self.wordIndices[tokens[i]]] += 1;
        wordDict_temp = PCA(n_components = self.pca_dim).fit_transform(wordDict_temp);
        for row_ind in range(np.shape(wordDict_temp)[0]):
            wordDict_temp[row_ind]/=np.linalg.norm(wordDict_temp[row_ind,:]) 
        for key in self.wordIndices.keys():
            self.wordDict[key] = wordDict_temp[self.wordIndices[key],:]
        return
    def saveFoodVectors(self, destinationFolder):
        path = os.path.join(destinationFolder, 'foodDict.pickle')
        with open(path, 'wb') as openedFile:
            pk.dump(self.foodDict, openedFile);openedFile.close();
        path = os.path.join(destinationFolder, 'wordDict.pickle')
        with open(path, 'wb') as openedFile:
            pk.dump(self.wordDict, openedFile);openedFile.close();
        path = os.path.join(destinationFolder, 'foodIndices.pickle')
        with open(path, 'wb') as openedFile:
            pk.dump(self.foodIndices, openedFile);openedFile.close();
        path = os.path.join(destinationFolder, 'wordIndices.pickle')
        with open(path, 'wb') as openedFile:
            pk.dump(self.wordIndices, openedFile);openedFile.close();
        print('Save Successful');
        return
#class deleteFoodItems(object):
    #Keep the indices as same because user history is saved wrt those indices only

#User Functionalities
def validUser(userID, userIDsURL): 
    with open(userIDsURL, 'rb') as ofile:
        allUserIDs = pk.load(ofile);
        ofile.close(); 
    if userID in allUserIDs:
        return True;
    else:
        return False;
def addUser(userID, userIDsURL, userHistoryURL): #Eg: addUser('vipin12_', 'userIDs.pickle')
    if validUser(userID, userIDsURL):
        raise Exception('User already exists');
    else:
        #Get all userIDs
        with open(userIDsURL, 'rb') as ofile:
            allUserIDs = pk.load(ofile);
            ofile.close();
        #Add this new user and a pickle file of his history
        allUserIDs+=[userID]
        
        with open(os.path.join(userHistoryURL,userID+'.pickle'), 'wb') as opfile:
            pk.dump({'foodIndices':[],'n_foodItemLogs':[]},opfile); opfile.close();
        #Save UsersIDs List
        with open(userIDsURL, 'wb') as ofile:
            pk.dump(allUserIDs, ofile); ofile.close();
        #Print all current users
        for id in allUserIDs:
            print(id);
    return
def deleteUser(userID, userIDsURL): #Eg: deleteUser('vipin12_', 'userIDs.pickle')
    if not validUser(userID, userIDsURL):
        raise Exception('User doesnt exists');
    else:
        #Get all userIDs
        with open(userIDsURL, 'rb') as ofile:
            allUserIDs = pk.load(ofile);
            ofile.close();
        #Delete this new user and the history saved so far
        allUserIDs.remove(userID)
        os.remove(os.path.join(userHistoryURL,userID+'.pickle'))
        #Save UsersIDs List
        with open(userIDsURL, 'wb') as ofile:
            pk.dump(allUserIDs, ofile); ofile.close();
        #Print all current users
        for id in allUserIDs:
            print(id);
    return
def addUserFoodLog(userID, foodNames):
    if not validUser(userID, userIDsURL):
        raise Exception('User doesnt exists');
    else: 
        #Open the users' food items and 
    
    
        

#        self.sCorrection = DL_distanceBased();
#        # Use self.sCorrection.spellCorrect(STRING_INPUT, trueWordSpellings);
#        # Make word Indices and do PCA
#
#
#    ############################################################
#    def addUser(self, userID): #Eg: .addUser('yourName')
#        if userID in self.users.keys():
#            raise Exception('User already exists');
#        self.users[userID] = np.ones((self.pca_dim,));
#        return;
#    def deleteUser(self, userID):
#        if userID not in self.users.keys():
#            raise Exception('userID not found!!')
#        del self.users[userID]
#    def addFoodItems(self, userID, histList): #Eg: .addFoodItems('yourName',['apple shake','egg omelette'])
#        if not isinstance(histList, list): #True if allW is just 1 string
#            raise Exception("input to addFoodItems() is usedID and a LIST of text logs");
#        #Preprocessing
#        for ind, text in enumerate(histList): 
#            histList[ind]  = text.replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(',', ' ').lower().strip()
#        histList = self.basic_process_text(histList)         
#        #Adding Vectors
#        for item in histList:
#            histList_Vec = np.zeros((self.pca_dim,));    
#            histList_cnt = 0;
#            try:
#                histList_Vec+=self.foodDict[item.lower().strip()]; histList_cnt+=1;
#            except Exception as e:
#                print('Food item not found: ', e); continue;  
#        if histList_cnt==0:
#            return;
#        else:
#            if (self.users[userID] == np.ones((self.pca_dim,))).all(): #First time logging
#               self.users[userID] = histList_Vec/histList_cnt; 
#            else: #Not a first time logging but second or further loggings
#                self.users[userID]*=self.users_logCount[userID];
#                self.users[userID]+=histList_Vec; 
#                self.users_logCount[userID]+=histList_cnt;
#                self.users[userID]/=self.users_logCount;
#        return 
#    ############################################################
#    def bestOfTheBest(self, userID, relevant_foodItems, top_k, also_add_irreevant_foodItems=False):
#        if len(relevant_foodItems)>=1:
#            if len(self.suggestions)>=top_k:
#                return;
#            relevant_foods_num = top_k-len(self.suggestions);
#            relevant_simScore = {}; 
#            for foodItem in (relevant_foodItems):
#                relevant_simScore[foodItem] = cosine_similarity(self.foodDict[foodItem], self.users[userID])[0][0]
#            relevant_foods_num = np.min( [relevant_foods_num, len(relevant_simScore.keys())] );
#            self.suggestions+=sorted(relevant_simScore, key=relevant_simScore.get, reverse=True)[:relevant_foods_num]
#        if also_add_irreevant_foodItems:
#            others_foods_num = top_k-len(self.suggestions);
#            others_simScore = {};
#            for foodItem in self.foodDict.keys():
#                if foodItem not in relevant_foodItems and foodItem not in self.suggestions:
#                    others_simScore[foodItem] = cosine_similarity(self.foodDict[foodItem], self.users[userID])[0][0]
#            self.suggestions+=sorted(others_simScore, key=others_simScore.get, reverse=True)[:others_foods_num]                    
#        return
#    def suggestFoods(self, userID, inputString, top_k):
#        if inputString==[]:
#            raise Exception('Input cannot be NULL!!')
#        # Note regarding Spell Correction of input string:
#        # Not always a spell correction is required!!
#        # Eg: inputString: 'ome' and he/she is typing 'omelette' and is in midway
#        # Eg: inputString: 'om' and he/she is typing a mistakeful 'omlet' and is in midway
#        # Else, fetch all strings with 'ome' present in it and find similarity score with self.users[self.usersID]
#        # OtherEg: 'ome egg' and user means 'egg omelette'
#        self.suggestions = [];
#        inputString = self.basic_process_text([inputString])[0]; #[] used only if inp is one item
#        inputString_tokens = wordpunct_tokenize(inputString)
#        # Partially or fully, all tokens are present in the beginning of at least one word in wordDict
#        all_tokens_valid = True; 
#        for token in inputString_tokens:
#            this_token_valid = False;
#            for eachWord in self.wordDict.keys():
#                if token in eachWord[:len(token)]:
#                    this_token_valid = True; continue;
#            all_tokens_valid = (all_tokens_valid and this_token_valid)
#        # Step 1: One or more words from inputString are matching one or more words from foodItem
#        if (all_tokens_valid and len(inputString_tokens)>=2 and len(self.suggestions)<top_k):
#            relevant_foodItems = [];
#            relevant_fully_foodItems =[];
#            for key in self.foodDict.keys():
#                key_tokens = wordpunct_tokenize(key);
#                for itoken in inputString_tokens:
#                    marked_presence = 0;
#                    for ktoken in key_tokens:
#                        if itoken in ktoken[:len(itoken)]:
#                            marked_presence +=1; continue;
#                    if marked_presence==len(inputString_tokens) and key not in relevant_fully_foodItems:
#                        relevant_fully_foodItems.append(key);
#                    elif marked_presence>0 and key not in relevant_foodItems:
#                        relevant_foodItems.append(key);
#            print('No.of Relevant Searches(Type 1): ', len(relevant_foodItems)+len(relevant_fully_foodItems));
#            self.bestOfTheBest(userID, relevant_fully_foodItems, top_k, also_add_irreevant_foodItems=False);
#            self.bestOfTheBest(userID, relevant_foodItems, top_k, also_add_irreevant_foodItems=False);
#        elif (all_tokens_valid and len(inputString_tokens)==1 and len(self.suggestions)<top_k): 
#            relevant_foodItems = [];
#            for key in self.foodDict.keys():
#                key_tokens = wordpunct_tokenize(key);
#                for itoken in inputString_tokens:
#                    marked_presence = 0;
#                    for ktoken in key_tokens:
#                        if itoken in ktoken[:len(itoken)]:
#                            marked_presence +=1; continue;
#                    if marked_presence>0 and key not in relevant_foodItems:
#                        relevant_foodItems.append(key);
#            print('No.of Relevant Searches(Type 2): ', len(relevant_foodItems));
#            self.bestOfTheBest(userID, relevant_foodItems, top_k, also_add_irreevant_foodItems=False);
#        # Step 2: If a token is not in our dictionary, find its closest word
#        if (len(self.suggestions)<top_k):
#            relevant_foodItems = [];
#            for token in inputString_tokens:
#                if token not in self.wordDict.keys():
#                    correctedToken = self.sCorrection.spellCorrect(token);
#                    for foodItem in self.foodDict.keys():
#                        if correctedToken in foodItem and foodItem not in relevant_foodItems and foodItem not in self.suggestions:
#                            relevant_foodItems.append(foodItem);
#            print('No.of Relevant Searches(Type 3): ', len(relevant_foodItems));
#            self.bestOfTheBest(userID, relevant_foodItems, top_k, also_add_irreevant_foodItems=True);
#        return self.suggestions;
#        ############################################################
        
#if __name__=="__main__":
#    #from SmartSuggestions import smartSuggestions
#    import time
#    ss = smartSuggestions(); ss.loadDictionaries(r'./data/testFoods1.txt');
#    ss.addUser('vipin');
#    givenFoods = [
#        r'Chana masala',
#        r'Chhole chawal',
#        r'rajma masala',
#        r'Kadhi Chawal',
#        r'Shahi paneer',
#        r'Tandoori Chicken Curry',
#        r'Mughlai Fish Curry',
#        r'Dal Tadka',
#        r'Spanish style omelette',
#        r'Bengali Mixed Vegetable Sabzi',
#        r'matar pulao',
#        r'daily special raita',
#        r'Kadhi pakoda',
#        r'samosa',
#        r'Kadak chai',
#        r'Ilaichi chai',
#        r'Adrak chai',
#        r'Gajar halwa',
#        r'Gajar halwa',
#        r'Hyderabadi Rice Kheer',
#        r'Scrambles eggs',
#        r'Minty Paneer Biryani',
#        r'butter chicken',
#        r'chicken biryani',
#        r'Lachha Parantha',
#        r'Mysore masala dosa',
#        r'Methi Alu Paranthas',
#        r'Aloo paneer',
#        r'Tawa Parantha',
#        r'Dahi bhalla',
#        r'Bhalla Papdi',
#        r'PURI WITH SABJI',
#        r'Mirch pakoda',
#        r'Quick Rawa Idli']
#    ss.addFoodItems('vipin',givenFoods);  
#    searches = ['aloo', 'alu sabzi', 'paneer', 'dosa', 'sambhar', 'kadi', 'dal', 'halw', 'fsh', 'adrak',
#                'cai', 'reita', 'tandry', 'mugle', 'tadk', 'ome']
#    searches = ['mattr']
#    results_vipin = {}
#    strt = time.time();
#    for searchItem in searches:
#        results_vipin[searchItem] = ss.suggestFoods('vipin',searchItem,10);
#    ed = time.time();
#    print('Avg Time in secs: ',(ed-strt)/len(searches));  del strt, ed, searchItem;
#
#    ss.addUser('sarika');
#    givenFoods = [
#        r'Banana Milkshake',
#        r'Apple Milkshake',
#        r'Strawberry Milkshake without Sugar',
#        r'Multi Grain Bread',
#        r'Scrambled egg',
#        r'Corn n peas salad, Subway',
#        r'Green peas patty, Subway',
#        r'Oven roasted chicken, Subway',
#        r'Aloo Patty, Subway',
#        r'Chickpeas and Olives dip',
#        r'Corriander Rice',
#        r'spinach rice',
#        r'Cucumber Greek Yogurt Salad',
#        r'Cucumber Raita',
#        r'Cabbage n dal parantha',
#        r'Kaju Matar Masala',
#        r'Bread dahi vada',
#        r'Gajar halwa',
#        r'Chai garam special',
#        r'Jasmine green Tea',
#        r'Papaya Cabbage Bean Sprouts salad']
#    ss.addFoodItems('sarika',givenFoods);  
#    searches = ['banana', 'scarbled', 'flavored', 'yogut', 'tea', 'haluwaa',
#                'omelette', 'gaajr', 'chai', 'patty', 'aloo', 'gobi', 'peas subway', 'roti']
#    results_sarika = {}
#    strt = time.time();
#    for searchItem in searches:
#        results_sarika[searchItem] = ss.suggestFoods('vipin',searchItem,10);
#    ed = time.time();
#    print('Avg Time in secs: ',(ed-strt)/len(searches));  del strt, ed, searchItem;
##    thefile = open('searches-sarika.txt', 'a')    
##    for key in results_sarika:  
##        thefile.write(key.upper().encode('utf-8').strip())
##        thefile.write("\n")
##        thefile.write("\n".join(results_sarika[key]).encode('utf-8').strip())
##        thefile.write("\n")
##        thefile.write("******************************************")
##        thefile.write("\n")
##    thefile.close()
##    del key, thefile
        
        
        
        
        
        
        
        
        
        
        
        
# Randomly pick few foodVecs and their average and compute similar to average foods!!
#def similarFoods(thisVec, foodDict, top_k):
#    try:
#        simScore = {};
#        for key in (foodDict.keys()):
#            simScore[key] = cosine_similarity(foodDict[key], thisVec)[0][0]
#        del thisVec, key
#    except Exception as e:
#        print(e)    
#    simFoods = sorted(simScore, key=simScore.get, reverse=True)[:top_k]  
#    for food in simFoods:
#        print(food)
#    return simScore
#for _ in range(10):
#    import random
#    randInds = random.sample(range(0, len(foodDict.keys())), 1)
#    sampleVec = np.zeros((len(foodDict[list(foodDict.keys())[0]]),))
#    for ind in randInds:
#        print(list(foodDict.keys())[ind]);
#        sampleVec+=(foodDict[list(foodDict.keys())[ind]])
#    sampleVec/=len(randInds)
#    print('*****************')
#    simScore = similarFoods(sampleVec, foodDict, 10)   
    
from scipy import sparse    
A = np.array([[1,2,0],[0,0,3],[1,0,4]])    
sA = sparse.csr_matrix(A)    
    
    