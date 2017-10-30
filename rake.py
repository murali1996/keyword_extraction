# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:32:28 2017

@author: Sai Muralidhar Jayanthi
Notes(Usage)
----
import rake
r = rake.Rake(r'C:\Users\Lenovo\Documents\Python Scripts\rawText1.txt')
text = r.text
sents = r.preprocess()
stopTerms = r.get_stopTerms()
candidates, allWords, allUniqueWords = r.get_candidates()
pText = r.get_pText()
co_occurance_graph = r.get_courMatrix()
#print(co_occurance_graph['food']['forest'],co_occurance_graph['food']['security'])
freq = r.get_freq()
scores = r.get_scores()
"""
import re, string, pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from collections import Counter, defaultdict
from itertools import product

class Rake(object):
    
    def __init__(self, path, word_min_len=1, candidate_max_len=5, window_size=5):
        #Input
        self.text = []
        try:
            with open(path,'r') as file:
                self.text = file.read()
        except Exception as e:
            print(e)
        self.word_min_len = word_min_len
        self.candidate_max_len = candidate_max_len
        if (window_size%2) is 0:
            raise Exception('Specify an odd-sized window!')
        else:
            self.window = window_size
        #Others1
        self.stopTerms = []
        #Others2
        self.sents = []
        self.candidates = []
        self.allWords = []
        self.allUniqueWords = []
        self.pText = []
        #Others3
        self.freq = []
        self.degree = []
        self.vecs = []
        self.co_occurance_graph = []
        self.scores = []
        #Check packages requirement
        self.check_packages()
        #Get the delimiting termms
        self.get_stopTerms()
        return
    def check_packages(self):
        try:
            import re, string, pandas as pd
            from nltk.corpus import stopwords
            from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
            from collections import Counter, defaultdict
            from itertools import product
        except Exception as e:
            print(e, "To install a <pkg>, open cmd and type 'pip install <pkg>' ")
        return
    def get_stopTerms(self):
        #Collect terms to Ignore
        ls1 = [str(item) for item in list(stopwords.words('english'))]
        ls2 = list(string.punctuation)
        self.stopTerms = list(ls1+ls2)
        return self.stopTerms 
    def preprocess(self):
        #Copying the original text
        ptext = self.text 
        #Lower-casing and removing html tags
        ptext = ptext.lower().strip()
        ptext = re.sub('<[^<]+?>', '', ptext)
        #Replacing '-' with ' ' ex.: "mind-blowing" --> "mind blowing"   
        ptext = re.sub('-',' ',ptext)
        #Converting the text to sentences and tokenizing
        self.sents  = []
        for sent in sent_tokenize(ptext):
            tokens = wordpunct_tokenize(sent)
            tokens = [token for token in tokens if (len(token)>=self.word_min_len)]
            if (' '.join(tokens)) is not '':
                self.sents.append(' '.join(tokens))
        return self.sents  
    def get_candidates(self, typ=2):
        """
        typ=1
        Splits the phrase between two stopterms like as follows:
        sent: "compatibility linear equations"
        candidates: 'comaptibility', 'linear', 'equations'
        
        typ=2
        Doesn't split the phrases like abobe type
        """
        self.candidates = []
        self.allWords = []
        self.allUniqueWords = []
        for sent in self.sents:
            dummy = word_tokenize(sent)
            temp=[]
            if typ is 1:
                for token in dummy:
                    #If the token is a stopTerm
                    if token in self.stopTerms:
                        #Add every n-gram less than max_size to the possible candidate list
                        if len(temp)<=self.candidate_max_len: 
                            if (' '.join(temp)) not in self.candidates and (' '.join(temp)) is not '':
                                self.candidates.append(' '.join(temp))
                        else:
                            for sz in range(2,self.candidate_max_len+1):
                                for i in range(0,len(temp)-sz+1):
                                    if ' '.join(temp[i:i+sz]) not in self.candidates and ' '.join(temp[i:i+sz]) is not '':
                                        self.candidates.append(' '.join(temp[i:i+sz]))
                        temp = []
                    #If not, add the word to temp, allWords, allUniqueWords
                    else:                  
                        #Collect list of words between two stopTerms
                        temp.append(token)
                        #Add each term of temp to possible-candidates list
                        if token not in self.candidates and token is not '':
                            self.candidates.append(token) 
                        #Keep a record of collecting words
                        self.allWords.append(token)
                        if token not in self.allUniqueWords:
                            self.allUniqueWords.append(token) 
            if typ is 2:
                for token in dummy:
                    # If token is a stopTerm, terminate adding to temp and add temp to candidates
                    if token in self.stopTerms:
                        if temp!=[]:
                            if (" ".join(temp)) not in self.candidates:
                                self.candidates.append(" ".join(temp)) 
                            temp=[]
                    #Else add the word to allWords, allUniqueWords, temp
                    else:
                        self.allWords.append(token)
                        if token not in self.allUniqueWords:
                            self.allUniqueWords.append(token)
                        temp.append(token)
                        if len(temp) is self.candidate_max_len:
                            if (" ".join(temp)) not in self.candidates:
                                self.candidates.append( (" ".join(temp)) )
                            temp = []
                            temp.append(token) 
                        else:
                            if (" ".join(temp)) not in self.candidates:
                                self.candidates.append( (" ".join(temp)) )            
        return self.candidates, self.allWords, self.allUniqueWords
    def get_pText(self):
        self.pText = word_tokenize( ' '.join(self.allWords) )
        return self.pText
    def get_courMatrix(self): #Co-occurance Matrix as a list
        self.co_occurance_graph = defaultdict(lambda: defaultdict(lambda: 0))
        move = int(self.window/2)
        for sent in self.sents:
            sent = word_tokenize(sent)
            lenSent = len(sent)
            for i in range(lenSent):
                for j in range(max(0,i-move),min(i+move,lenSent-1)+1):
                    self.co_occurance_graph[sent[i]][sent[j]]+=1
        return self.co_occurance_graph
    def get_freq(self):
        ctr = Counter(self.allWords) #freq.most_common(10) #freq.has_key('food') #freq.get('food')
        self.freq = []
        for word in self.allUniqueWords:
            self.freq.append(ctr.get(word))    
        return self.freq
    def get_degree(self):
        self.get_courMatrix()
        self.degree = []
        for word in self.allUniqueWords:
            degree = 0
            for coword in self.allUniqueWords:
                if word is not coword:
                    degree+=self.co_occurance_graph[word][coword]
            self.degree.append(degree)
        return self.degree
    def get_unit_vecs(self):
        self.vecs = []
        if len(self.freq)!=len(self.degree):
            raise Exception("List Lengths Not Compatible. Error!")
        for i in range(len(self.freq)):
            num = float(self.freq[i])
            den = num+float(self.degree[i])
            self.vecs.append(num/den)
        return self.vecs
    def get_ranks(self,type=2):
        self.scores = []
        for cd in self.candidates:
            thisScore=0.0
            cd = word_tokenize(cd)
            #Remove duplicate words from the candidate phrase
            dummy = []
            for tempo in cd:
                if tempo not in dummy:
                    dummy.append(tempo)
            cd = dummy
            if type is 1:
                pass
            elif type is 2:
                #Compute sum of unit_vecs of each word in each candidate
                for word in cd:
                    thisScore+= self.vecs[self.allUniqueWords.index(word)]
                self.scores.append(thisScore)
        #Return a dictionary of all candidates and their scores
        self.scores = pd.DataFrame(dict({'phrase':self.candidates,'score':self.scores}))
        self.scores.sort_values(by=['score'], ascending=False, inplace=True)
        return self.scores


if __name__=="__main__":
    import rake
    r = rake.Rake(r'C:\Users\Lenovo\Documents\Python Scripts\rawText4.txt')
    text = r.text
    sents = r.preprocess()
    stopTerms = r.get_stopTerms()
    candidates, allWords, allUniqueWords = r.get_candidates()
    #pText = r.get_pText()
    co_occurance_graph = r.get_courMatrix()
    #print(co_occurance_graph['food']['forest'],co_occurance_graph['food']['security'])
    freq = r.get_freq()
    degree = r.get_degree()
    vecs = r.get_unit_vecs()
    ranks = r.get_ranks()





"""
#In get_scores() method
for i in range(0,cdLen):
    for j in range(i+1,cdLen):
        if i is not j:
            score*=(1+(float(self.co_occurance_graph[cd[i]][cd[j]])/float(self.freq.get(cd[i]))))
all_scores.append(score)        
"""