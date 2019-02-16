"""
Title: Lab1
Author: Prem Chotepanit
Registration number: 180127900
OS: Window10
Python version: 3.6.8
"""

import sys,os,random,regex as re
#import multiprocessing as mp
import matplotlib.pyplot as plt
random.seed(1)
dir_name = sys.argv[1]
print(dir_name)
Path = os.path.dirname((os.path.realpath(__file__)))\
    +'\\'+dir_name\
    +'\\'+"txt_sentoken"
    #Accessing folder "review_palarity"
Class = os.listdir(Path)
"""
Part 1: Choosing training data and testing data 
- using random 800 file names for training data and 200 for testing data
"""
#Training Data
Training_pos_data = []
Training_neg_data = []


"""
Firstly, the program randoms a file with its path for 800 times 
for both the positive class.
Then, it appends them to training data.
"""
text_class = 'pos' #class positive
pos_path = Path + '\\' + text_class
pos_files = set(os.listdir(pos_path))
Training_pos_data = set(random.sample(pos_files, 800))
Testing_pos_data = pos_files - Training_pos_data #filtering training data from total files
"""
Next, the program does the same process with negative files
"""

text_class = 'neg' #class negative
neg_path = Path + '\\' + text_class
neg_files = set(os.listdir(neg_path))
Training_neg_data = set(random.sample(neg_files, 800))
Testing_neg_data = neg_files - Training_neg_data #filtering training data from total files

"""
Part 2: Define the function which returns bag of words from
an input file.
"""
def Unigram_Preprocess(File): #Unigram
    if(File in pos_files):
        f = open(pos_path +'\\'+ File, "r")
    else:
        f = open(neg_path +'\\'+ File, "r")
    text = f.read()
    output = {}
    text = re.sub("[^\w']"," ",text).split()
    for word in text:
	    if word in output:
	        output[word] += 1
	    else:
	        output[word] = 1
    return output
## Input: a text file's name with path
## Output: {"there's": 1, 'a': 10, 'scene': 3, 'early': 1,...


"""
Part3: processing data
The program define a function to preprocess data by generating a bag of word from file name.

X_train_raw: Input data as filenames
W: the weight vector which is initialled as NULL
y: the class of a document: -1 for negative; 1 for positive
Precision: collection of precision from each process
Recall: collection of recell from each process
iteration: number of processing times


The function, then, read the files to processes the updating W using data.
Finally, the program average W and return W,recall,precision as a dictionary.
"""

def preprocess_Data(X_train_raw,Feature ='unigram'):
    D = []
    for x in X_train_raw:
        d = {}
        d['y'] = x in pos_files #Get label of each document
        if Feature == 'unigram':
            d['Phi_X'] = Unigram_Preprocess(x) #Get a bag of word  of each document
        D.append(d)
    return D
'''
The function perception is meant to calculate W and learning process.
'''
def perceptron(D):
    W ={}
    Precision = []
    Recall = []
    Accuracy = []
    iteration = 5
    
    '''
    process each iteration
    '''
    for i in range(iteration):
        '''
        Process updating w for each iteration and count parameters to 
        calculate recall and precision
        '''
        True_Positive = 0 #initial True positive
        False_Positive = 0 #initial False positive
        False_Negative = 0 #initial False negative
        True_Negative = 0 #initial True negative
        random.shuffle(D) #shuffle Data
        j = 0
        '''
        process each document
        '''
        for d in D:
            percentage = "Processing "+"%f"%((i+(j/len(D)))\
               /(iteration)*100)+"%              "
            print(percentage, end ="\r" ) #Printing percentage
            j += 1
            y = d['y'] #Getting a lable of a document
            Phi_X = d['Phi_X'] #Making a bag of words
            w_Phi_X = sum([W[key]*Phi_X[key] \
                for key in W if key in Phi_X]) #Multiply 2 vector
            y = 2*int(y)-1 #Normalise Y of boolean to -1 or 1
            sign_w_Phi_x = 2*int(w_Phi_X > 0) -1
            if y!=sign_w_Phi_x :
                if sign_w_Phi_x  == 1:
                    False_Positive += 1 #False prediction as positive
                else:
                    False_Negative += 1 #False prediction as negative
                for key in Phi_X.keys(): 
                    if key in W: #check whether a key exist
                        W[key] += y*Phi_X[key] #adding a term value
                    else:
                        W[key] = y*Phi_X[key] #updating a term key
            else:
                if y == 1:
                    True_Positive += 1 #Counting true positive
                else:
                    True_Negative += 1 #Counting true negative
        Recall.append(True_Positive/(True_Positive+False_Negative))
        Precision.append(True_Positive/(True_Positive+False_Positive))
        Accuracy.append((True_Negative+True_Positive)/\
            (True_Negative+True_Positive+False_Negative+False_Positive))
    C = len(D)
    W = {key: value/C for (key,value) in W.items()} # Averaging W
    print("Done!                            ")
    return {"W": W\
            ,"Recall":Recall\
            ,"Precision":Precision\
            ,"Accuracy":Accuracy}


X_train_Unigram = list(Training_pos_data.union(Training_neg_data)) #Combine set of file names

'''
Part4 : Add the testing function
'''
def test_Data(W,Feature = 'Unigram'):
    X_test = list(Testing_pos_data.union(Testing_neg_data))
    True_Positive = 0 #initial True positive
    False_Positive = 0 #initial False positive
    False_Negative = 0 #initial False negative
    True_Negative = 0 #initial True negative
    if Feature == 'Unigram':
        D_test = preprocess_Data(X_test)
    random.shuffle(D_test)
    for d in D_test:
        y = d['y'] #Getting a lable of a document
        Phi_X = d['Phi_X'] #Making a bag of words
        w_Phi_X = sum([W[key]*Phi_X[key] \
        for key in W if key in Phi_X]) #Multiply 2 vector
        y = 2*int(y)-1 #Normalise Y of boolean to -1 or 1
        sign_w_Phi_x = 2*int(w_Phi_X > 0) -1
        if y!=sign_w_Phi_x :
            if sign_w_Phi_x  == 1:
                False_Positive += 1 #False prediction as positive
            else:
                False_Negative += 1 #False prediction as negative
        else:
            if sign_w_Phi_x == 1:
                True_Positive += 1 #Correct prediction as positive
            else:
                True_Negative += 1 #Correct prediction as negative
        
    Recall = True_Positive/(True_Positive+False_Negative)
    Precision = True_Positive/(True_Positive+False_Positive)
    return (Recall,Precision)
'''
Initail processing of each feature pararelly
'''
def Unigram_Perceptron(): #for unigram
    D = preprocess_Data(X_train_Unigram) #Preprocess
    Unigram = perceptron(D) #Process perceptron
    Recall = Unigram['Recall']
    Precision = Unigram['Precision']
    Accuracy = Unigram['Accuracy']
    F1_Score = [2 * (Recall[i]*Precision[i]) / (Precision[i]+Recall[i])\
        for i in range(len(Recall))]
    Recall_test,Precision_test = test_Data(Unigram['W'])
    print("The Recall of unigram for testing data is "\
         + str(Recall_test))
    print("The Precision of unigram for testing data is "\
        + str(Precision_test))
    '''
    Ploting evaluation
    '''
    plt.plot(range(len(Recall)),Recall,'r',label = 'Recall')
    plt.plot(range(len(Precision)),Precision,'b',label = 'Precision')
    plt.plot(range(len(F1_Score)),F1_Score,'g',label = 'F1_Score')
    plt.plot(range(len(Accuracy)),Accuracy,color='black',label = 'Accuracy')
    plt.xlabel('iteration')
    plt.title('Unigram learning process')
    plt.legend()
    plt.show()

'''
A function for process bigram beyond bag of words
'''
def Bigram_Perceptron():
    D = preprocess_Data(X_train_Unigram) #Preprocess
    Unigram = perceptron(D) #Process perceptron
    Recall = Unigram['Recall']
    Precision = Unigram['Precision']
    Accuracy = Unigram['Accuracy']
    F1_Score = [2 * (Recall[i]*Precision[i]) / (Precision[i]+Recall[i])\
        for i in range(len(Recall))]
    '''
    Ploting evaluation
    '''
    plt.plot(range(len(Recall)),Recall,'r',label = 'Recall')
    plt.plot(range(len(Precision)),Precision,'b',label = 'Precision')
    plt.plot(range(len(F1_Score)),F1_Score,'g',label = 'F1_Score')
    plt.plot(range(len(Accuracy)),Accuracy,color='black',label = 'Accuracy')
    plt.xlabel('iteration')
    plt.title('Unigram learning process')
    plt.legend()
    plt.show()


Unigram_Perceptron()



