# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:38:56 2018
@author: murali.sai
"""
#==============================================================================
# Files:
# insights.py  <-----> makeDictionary.py  <------> scrapeFoodData.py
#              <-----> spellCorrection.py
#==============================================================================

from makeDictionary import makeDictionary, process_text
from nltk.tokenize import wordpunct_tokenize
from spellCorrection import DL_distanceBased
import nltk
import re
############################################################################
# Sample Tex Logs
textLogs = [r'Cofee  no sugar skim mlk (2 ups)',
            r'2small muli grain tost and a boioed egg',
            r'parta saij',
            r'chapati crry',
            r'friut slaad an 2eggs omlette',
            r'Ate half appple with spme parthha ad spruots;',
            r' drnk 500mL mixd reel jiuce and hd 2 rostd brd slicces with muton tiki',
            r'and paneerr vegs biriyan with chat']
textLogs = [r'patra sbji',
            r'friut slaad an 2eggs omlette'];
textLogs = [r'gongora']
# Pre-processing: Remove all punctuations: Only alphabets and numerals can be seen 
textLogs = process_text(textLogs);
#############################################################################

individualNames, fullNames = makeDictionary();
# Spell Correction Layer-1
for ind, log in enumerate(textLogs):
    #Assumption from hereon is that the logs can ONLY contain
    #lowercase_alphabets, '_', and 'numerals'
    dl = DL_distanceBased(individualNames, log); 
    textLogs[ind] = dl.spellCorrect();
    print(sum(dl.minScores));
# Spell Correction Layer-2
# Based on tags, group food items together with '_' and spell correct with fullNames
#for ind, log in enumerate(textLogs):
#    log_tags = nltk.pos_tag(wordpunct_tokenize(log));
#    log_tags_temp = [];
#    i = 0;
#    while i<len(log_tags):
#        while i<len(log_tags) and log_tags[i][1]!='NN' and log_tags[i][1]!='FW':
#            log_tags_temp.append(log_tags[i][0]);
#            i+=1
#        dummy = r'';
#        while(i<len(log_tags) and (log_tags[i][1]=='NN' or log_tags[i][1]=='FW')):
#            dummy+=(log_tags[i][0]+' ');
#            i+=1
#        if dummy!='':
#            dummy = dummy.strip().replace(' ','_');
#            log_tags_temp.append(dummy);
#    textLogs[ind] = ' '.join(log_tags_temp)
#for ind, log in enumerate(textLogs):
#    #Assumption from hereon is that the logs can ONLY contain
#    #lowercase_alphabets, '_', and 'numerals'
#    dl = DL_distanceBased(fullNames, log); 
#    textLogs[ind] = dl.spellCorrect();
#    print(sum(dl.minScores));
###############################################################################    
#Generate Some mistake-ful text terms from the food item names to check algo's precision
#...
#....
###############################################################################







