"""
Title: Lab1
Author: Prem Chotepanit
Registration number: 180127900
OS: Window10
Python version: 3.6.8
"""

import sys,os,random,regex as re
from numpy import sign
random.seed(777)
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
print(neg_path+'\\'+list(Testing_neg_data)[0])
"""
Part 2: Define the function which returns bag of words from
an input file.
"""
def Phi(File):
    f = open(File, "r")
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
The program preprocess data by generating a bag of word from file name.

W: the weight vector which is initialled as NULL
X_test: the bag of words each test file
Precision: collection of precision from each process
Recall: collection of recell from each process
iteration: number of processing times
"""

X_train_raw = list(Training_pos_data.union(Training_neg_data)) #Combine set of file names
W ={}
Precision = []
Recall = []
X_train = []
iteration =50

'''
process each iteration
'''
for i in range(iteration):
    True_Positive = 0 #initial True positive
    False_Positive = 0 #initial False positive
    False_Negative = 0 #initial False negative
    random.shuffle(X_train_raw) #shuffle X_train
    '''
    process each document
    '''
    for x in X_train_raw:
        y = x in pos_files #Getting a lable of a document
        '''
        Reading file and making a bag of words
        '''
        if y:
            X = pos_path+'\\'+x      
        else:
            X = neg_path+'\\'+x
        Phi_X = Phi(X) #Making a bag of words
        w_Phi_X = sum({key: W[key]*Phi_X[key] \
            for key in W if key in Phi_X}.values()) #Multiply 2 vector
        y = 2*int(y)-1 #Normalise Y of boolean to -1 or 1
        sign_w_Phi_x = sign(w_Phi_X)
        if y!=sign_w_Phi_x :
            if sign_w_Phi_x  == 1:
                False_Positive += 1 #False prediction as positive
            if sign_w_Phi_x  == -1:
                False_Negative += 1 #False prediction as negative
            for key in Phi_X.keys(): 
                if key in W: #check whether a key exist
                    W[key] += y*Phi_X[key] #adding a term value
                else:
                    W[key] = y*Phi_X[key] #updating a term key
        else:
            if y == 1:
                True_Positive += 1 #Counting true positive
    Recall.append(True_Positive/(True_Positive+False_Negative))
    Precision.append(True_Positive/(True_Positive+False_Positive))
print(Recall)
print(Precision)
    #print(str(False_Negative)+'\n')