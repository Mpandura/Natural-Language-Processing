'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
import re

from nltk.corpus import stopwords
from nltk.metrics import *
from math import ceil 
from nltk.collocations import *
from nltk.classify import MaxentClassifier

#import sentiment__read_subjectivity_words
# initialize the positive, neutral and negative word lists
#(positivelist, neutrallist, negativelist) = #sentiment__read_subjectivity_words.read_three_types()

# define a feature definition function here


# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr,limitTest):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  #lTest=int(limitTest)
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  t = open('./test.tsv','r')
  
  
  # ------ For Training --------
  
  
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines0
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
#  print('This is for train phrase list')
#  for phrase in phraselist[:10]:
#    print (phrase)
  
  def alpha_filter(w):
    pattern=re.compile('^[^a-z]+$')
    if(pattern.match(w)):
      return True
    else:
      return False
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  # add all the phrases
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  
  # print a few
 # print('This is for phrase docs')
  #for phrase in phrasedocs[:10]:
  #  print (phrase)

  # extracting just the words
  
  movieWords = []
  for w in phraselist:
    tokens = nltk.word_tokenize(w[0])
    for w1 in tokens:
        movieWords.append(w1)
  
  
  #Changing the case 
  movieWordsL=[w.lower() for w in movieWords]
  
  #Filter Punctuation Marks
  #Declare and Define Alpha Filter

  # Apply it to the MoviewordsLower case list
  
  movieWordsLP=[w for w in movieWordsL if not alpha_filter(w)]  
  # Stop words from nltk 
  stopwords= nltk.corpus.stopwords.words('english')
  
  #Filter based on stop  ------------------------------------------------change from L to LP
  movieWordsLPS= [w for w in movieWordsLP if not w in stopwords]
 
  #Stemming Using Porter Stemmer
  porter= nltk.PorterStemmer()
  movieWordsLPSPr=[porter.stem(t) for t in movieWordsLPS]  
  #movieWordsLPSPr=[porter.stem(t) for t in movieWordsLP]  
       
  
  #print the list
  #for word in movieWordsLPSPr:
  #  print (word)

  # possibly filter tokens
  all_words= nltk.FreqDist(movieWordsLPSPr)
  #word_items = all_words.most_common(50)
  #word_items = all_words.most_common(2500)
  word_items = all_words.most_common(5000)  
  common_set=[word for (word,freq) in word_items]
  
  #----------------PART II-------------------------
  # Document Features function
  
  #Define Read Subjectivity Function
  def readSubjectivity(path):
    flexicon = open(path, 'r')
  # initialize an empty dictionary
    sldict = { }
    weakpos=2
    weakneg=-2
    strongpos=4
    strongneg=-4
    for line in flexicon:
      fields = line.split()   # default is to split on whitespace
      # split each field on the '=' and keep the second part as the value
      strength = fields[0].split("=")[1]
      word = fields[2].split("=")[1]
      posTag = fields[3].split("=")[1]
      stemmed = fields[4].split("=")[1]
      polarity = fields[5].split("=")[1]
      if (stemmed == 'y'):
        isStemmed = True
      else:
        isStemmed = False
    # put a dictionary entry with the word as the keyword
    #     and a list of the other values
    #sldict[word] = [strength, posTag, isStemmed, polarity]
      if strength=='weaksubj' and polarity == 'positive':
        sldict[word] = [strength, posTag, isStemmed, polarity,weakpos]
      if strength=='weaksubj' and polarity == 'negative':
        sldict[word] = [strength, posTag, isStemmed, polarity,weakneg]
      if strength=='strongsubj' and polarity == 'positive':
        sldict[word] = [strength, posTag, isStemmed, polarity,strongpos]
      if strength=='strongsubj' and polarity == 'negative':
        sldict[word] = [strength, posTag, isStemmed, polarity,strongneg]
      
    
    # Returning scores based on polarity and strength
    
    
    return sldict
  #Enter the path of subjclueslen1-HLTEMNLP05 relative to the PC
  SLPath="E:/myFinalProject/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
  SL= readSubjectivity(SLPath)

    
  def movieReviews_polarity(document,common_set,SL):
    single_doc=set(document)
    features={}
    score=0
    final_score=0
    
    #tokens = nltk.word_tokenize(document)
    
    len_doc=len(tokens)
    max_score=(len_doc*4)
    min_score=-(len_doc*4)
    neutral=(max_score+min_score)/2
    range_helper=(2*max_score)/10
    for x in document:
      if x in SL:
        strength,posTag,isStemmed,polarity,score1=SL[x]
        score=score+score1
    if(score >= (-5*range_helper) and score< (-3*range_helper)):
      final_score=0
      #features['strong negative']=score
    if(score >= (-3*range_helper) and score< (-1*range_helper)):
      final_score=1
      #features['negative']=score
    if(score >= (-1*range_helper) and score< (1*range_helper)):
      final_score=2
      #features['neutral']=score
    if(score >= (1*range_helper) and score< (3*range_helper)):
      final_score=3
      #features['positive']=score
    if(score >= (3*range_helper) and score< (5*range_helper)):
      final_score=4
      #features['strong positive']=score
    
    #score2=''
    #score2=score
    features['Score']=score
    return features

            
  #Final Feature Sets
  featuresets= [(movieReviews_polarity(d,common_set,SL),c) for (d,c) in phrasedocs]
  
  #for f in featuresets:
  #  print(f)
  trainl= ceil (0.7*limit)
  testl=ceil(0.3*limit)
  train_set,test_set=featuresets[trainl:],featuresets[:testl]
  classifier=nltk.NaiveBayesClassifier.train(train_set)
  #print(nltk.classify.accuracy(classifier,test_set))
  print('------------------------------------------------------------------------------------------')
  print('Naive Bayes Classifier Accuracy for Training/Test Set given :',nltk.classify.accuracy(classifier,test_set))
  print('------------------------------------------------------------------------------------------') 
  
  
 # print('------------------SCI-KIT-LEARN -GENERATING CSV FILE ---------------------------------- ')
  
  def writeFeatureSets(featuresets1, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets1[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string 
    #   for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets1:
        featureline = ''
        for key in featurenames:
            featureline += str(featureset[0][key]) + ','
        featureline += str(featureset[1])
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()

  
  #for (features, label) in train_set:
  #  print(classifier.classify(features),label)
  
  #Cross Validation , Recall , F-Measure, Precision
  
  #Eval Measures Function
  def eval_measures(reflist, testlist):
    #initialize sets
    Final_Precision=[]
    Final_Recall=[]
    refspos = set()
    refsneg = set()
    refpos = set()
    refneg = set()
    refneutral= set()
    testspos = set()
    testsneg = set()
    testpos = set()
    testneg = set()
    testneutral= set()
    # get gold labels
    for j, label in enumerate(reflist):
      if label == 0: refsneg.add(j)
      if label == 1: refneg.add(j)
      if label == 2: refneutral.add(j)
      if label == 3: refpos.add(j)
      if label == 4: refspos.add(j)
    # get predicted labels
    for k, label in enumerate(testlist):
      if label == 0: testsneg.add(k)
      if label == 1: testneg.add(k)
      if label == 2: testneutral.add(k)
      if label == 3: testpos.add(k)
      if label == 4: testspos.add(k)
    #compute precision for all labels
    strongpos_precision = precision(refspos, testspos)
    strongneg_precision = precision(refsneg, testsneg)
    Neutral_precision = precision(refneutral, testneutral)
    Pos_precision = precision(refpos, testpos)
    Neg_precision = precision(refneg, testneg)
    Final_Precision.append(strongpos_precision)
    Final_Precision.append(strongneg_precision)
    Final_Precision.append(Neutral_precision)
    Final_Precision.append(Pos_precision)
    Final_Precision.append(Neg_precision)
  # Computer Recall for all labels
  
    strongpos_recall = recall(refspos, testspos)
    strongneg_recall = recall(refsneg, testsneg)
    Neutral_recall = recall(refneutral, testneutral)
    Pos_recall = recall(refpos, testpos)
    Neg_recall = recall(refneg, testneg)
    Final_Recall.append(strongpos_recall)
    Final_Recall.append(strongneg_recall)
    Final_Recall.append(Neutral_recall)
    Final_Recall.append(Pos_recall)
    Final_Recall.append(Neg_recall)
    return (Final_Precision,Final_Recall)
    #return (Pos_precision,Neg_precision,Pos_recall,Neg_recall)
  #Cross Validation Function
  def cross_validate_evaluate(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    pos_precision_list = []
    neg_precision_list = []
    #pos_precision_list.append(0)
    #neg_precision_list.append(0)
    Final_Precision=[]
    Final_Recall=[]
    j=0
    z=0
    sp_p_list=[]
    p_p_list=[]
    ne_p_list=[]
    n_p_list=[]
    sn_p_list=[]
    sp_r_list=[]
    p_r_list=[]
    ne_r_list=[]
    n_r_list=[]
    sn_r_list=[]
    sp_f_measure_mean=[]
    p_f_measure_mean=[]
    ne_f_measure_mean=[]
    n_f_measure_mean=[]
    sn_f_measure_mean=[]
    #  over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # build reference and test lists on the test set per round
        reflist = []
        testlist = []
        pos_recall_list=[]
        neg_recall_list=[]
        pos_recall_list.append(0)
        neg_recall_list.append(0)
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))
        # call the evaluation measures function
        Final_Precision,Final_Recall = eval_measures(reflist, testlist)
        #(pos_precision,neg_precision,pos_recall,neg_recall)=eval_measures(reflist,testlist)
        for fp1 in range(len(Final_Precision)):
            #print (fp1)
            if(Final_Precision[fp1]== None):
                Final_Precision[fp1]=0
            if (fp1 == 0):
                sp_p_list.append(Final_Precision[fp1])
            elif (fp1== 1):
                p_p_list.append(Final_Precision[fp1])
            elif (fp1== 2):
                ne_p_list.append(Final_Precision[fp1])
            elif (fp1== 3):
                n_p_list.append(Final_Precision[fp1])
            else:
                sn_p_list.append(Final_Precision[fp1])
            #print('FP',Final_Precision[fp1])         
        #vp=vp+Final_Precision[0]
        for fp2 in range(len(Final_Recall)):
            if(Final_Recall[fp2]==None):
                Final_Recall[fp2]=0
            if (fp2 == 0):
                sp_r_list.append(Final_Recall[fp2])
            elif (fp2== 1):
                p_r_list.append(Final_Recall[fp2])
            elif (fp2== 2):
                ne_r_list.append(Final_Recall[fp2])
            elif (fp2== 3):
                n_r_list.append(Final_Recall[fp2])
            else:
                sn_r_list.append(Final_Recall[fp2])
       
    print('Done with cross-validation')
    # find mean precision over all rounds
    print('---------------------------------------------------------------------------------------------')
    print('----------------------------<< CROSS VALIDATION >>-------------------------------------------')
    print('---------------------------------------------------------------------------------------------')
    print('')
    strong_mean_pos_precision = sum(sp_p_list)/num_folds
    mean_pos_precision=sum(p_p_list)/num_folds
    mean_neu_precision = sum(ne_p_list)/num_folds
    mean_neg_precision= sum(n_p_list)/num_folds
    strong_mean_neg_precision= sum(sn_p_list)/num_folds
    strong_mean_pos_recall = sum(sp_r_list)/num_folds
    mean_pos_recall=sum(p_r_list)/num_folds
    mean_neu_recall= sum(ne_r_list)/num_folds
    mean_neg_recall= sum(n_r_list)/num_folds
    strong_mean_neg_recall= sum(sn_r_list)/num_folds
    
    print('                  << PRECISION MEAN >>        << RECALL MEAN >>         << F-MEASURE >>')
    print('')
    if ((strong_mean_pos_precision+strong_mean_pos_recall) != 0 ):
      sp_f_measure_mean= (2* ((strong_mean_pos_precision * strong_mean_pos_recall)/ (strong_mean_pos_precision+strong_mean_pos_recall)))
      print('Strong Positive : ',strong_mean_pos_precision,'     ',strong_mean_pos_recall,'     ',sp_f_measure_mean)
      print('')
    else:
      print('Strong Positive : ',strong_mean_pos_precision,'     ',strong_mean_pos_recall,'     ','Fmeasure is 0')
    if ((mean_pos_precision+mean_pos_recall) != 0 ):
      p_f_measure_mean= (2* ((mean_pos_precision * mean_pos_recall)/ (mean_pos_precision+mean_pos_recall)))
      print(' Positive       : ',mean_pos_precision,'     ',mean_pos_recall,'     ',p_f_measure_mean)
      print('')
    else:
      print(' Positive       : ',mean_pos_precision,'     ',mean_pos_recall,'     ','Fmeasure is 0')
      print('')
    if ((mean_neu_precision+mean_neu_recall) != 0 ):
      ne_f_measure_mean= (2* ((mean_neu_precision * mean_neu_recall)/ (mean_neu_precision+mean_neu_recall)))
      print(' Neutral        : ',mean_neu_precision,'     ',mean_neu_recall,'     ',ne_f_measure_mean)
      print('')
    else:
      print(' Neutral        : ',mean_neu_precision,'     ',mean_neu_recall,'     ','Fmeasure is 0')
    if ((mean_neg_precision+mean_neg_recall) != 0 ):
      n_f_measure_mean= (2* ((mean_neg_precision * mean_neg_recall)/ (mean_neg_precision+mean_neg_recall)))
      print(' Negative       : ',mean_neg_precision,'     ',mean_neg_recall,'     ',n_f_measure_mean)
      print('')
    else:
      print(' Negative       : ',mean_neg_precision,'     ',mean_neg_recall,'     ','Fmeasure is 0')
    if ((strong_mean_neg_precision+strong_mean_neg_recall) != 0 ):
      sn_f_measure_mean= (2* ((strong_mean_neg_precision * strong_mean_neg_recall)/ (strong_mean_neg_precision+strong_mean_neg_recall)))
      print('Strong Negative : ',strong_mean_neg_precision,'     ',strong_mean_neg_recall,'     ',sn_f_measure_mean)
      print('')
    else:
      print('Strong Negative : ',strong_mean_neg_precision,'     ',strong_mean_neg_recall,'     ','Fmeasure is 0')
    print('---------------------------------------------------------------------------------------------')
     
     
    
  # Call The cross validation function using 3 folds 
  
  cross_validate_evaluate(3,featuresets)

  
  writeFeatureSets(featuresets,limitTest)
  print('> Feature sets transferred to CSV file')
  

  
  bigram_measures = nltk.collocations.BigramAssocMeasures()

  # create the bigram finder on the movie review words in sequence
  finder = BigramCollocationFinder.from_words(movieWords)

  # define the top 500 bigrams using the chi squared measure
  bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
  #bigram_features[:50]

  def bigram_document_features(document, common_set):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in common_set:
      features['contains({})'.format(word)] = (word in document_words)
    for bigram in bigram_features:
      features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

  #bigram_featuresets = [(bigram_document_features(d, common_set), c) for (d, c) in phrasedocs]  

  
  #train_set2, test_set2 = bigram_featuresets[trainl:], bigram_featuresets[:testl]
  #classifier2 = nltk.NaiveBayesClassifier.train(train_set2)
  #nltk.classify.accuracy(classifier2, test_set2)

  #print('-----------------------------------------------------------------------------------')
  #print('Bigram - Featuresets Naive Bayes Classifier Accuracy :',nltk.classify.accuracy(classifier2,test_set2))
  #print('-----------------------------------------------------------------------------------') 

  ## other classifiers ##

  #DecisionTree Classifier
  #classifier3 = nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff =0, support_cutoff =0)
  #nltk.classify.accuracy(classifier, test_set)
  #print('-----------------------------------------------------------------------------------')
  #print('Decision Tree Classifier Accuracy :',nltk.classify.accuracy(classifier3, test_set))
  #print('-----------------------------------------------------------------------------------\n') 

  #MaxEnt Classifier
   
  #classifier4 = nltk.MaxentClassifier.train(train_set, max_iter = 1)
  #nltk.classify.accuracy(classifier, test_set)
  #print('-----------------------------------------------------------------------------------')
  #print('Max Entropy Classifier Accuracy :',nltk.classify.accuracy(classifier4, test_set))
  #print('-----------------------------------------------------------------------------------\n') 

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print ('usage: classifyKaggle.py <corpus-dir> <limit> <Path for CSV file to transfer Featuresets>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2],sys.argv[3])