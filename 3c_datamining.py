## 0
import pandas as pd
import html

## 1

def read_file(filename):
    """ Reads a file. """
    
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    
    return lines


def create_dataframe(lines):
    """ Create a dataframe from a csv file. """
    
    # get column names from first line
    col_names = lines[0].split(';')
    cols = [col_names[i].strip() for i in range(len(col_names))]
    
    # prepare data frame
    amount_lines = len(lines)
    df = pd.DataFrame(columns=cols, index=range(amount_lines - 1))
    
    # fill dataframe
    i = 0
    for line in lines[1:]:

        parts = line.split(';', 1)

        df.loc[i].label = parts[0]
        df.loc[i].text = parts[1].strip()

        i = i + 1
        
    return df

## 2
lines = read_file("data/SmsCollection.csv")
df = create_dataframe(lines)

## 3
from nltk import stem
from nltk.corpus import stopwords

stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

## 4
def alternative_review_messages(msg):
    """ This is a function that converts messages to all lowercase letters, removes stopwords
        and stems all words: stemming reduces inflection forms to normalise words with the same lemma."""
    
    # unescape html
    msg = html.unescape(msg)
    
    # converting messages to lowercase
    msg = msg.lower()
        
    # removing stopwords
    msg = [word for word in msg.split() if word not in stopwords]
    
    # using a stemmer
    msg = " ".join([stemmer.stem(word) for word in msg])
    
    return msg

def lower_messages(msg):
    
    # unescape html
    msg = html.unescape(msg)
    
    # lowercase message
    msg = msg.lower()
    
    return msg

## 4.5
with pd.option_context('display.min_rows', 50, 'display.max_colwidth', 10000):
    display(df)

## 5
from sklearn.model_selection import train_test_split

# use term frequency - inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

def clean_and_split_data(df, clean_func, test_size):
    df_train = df.copy()
    df_train["text"] = df["text"].apply(clean_func)
    
    # split dataset into test and training
    trainmsg, testmsg, trainlabel, testlabel = train_test_split(
                df_train['text'], 
                df_train['label'], 
                test_size = test_size, 
                random_state = 1)
    
    return trainmsg, testmsg, trainlabel, testlabel

def train_model(trainmsg, trainlabel):
        
    # vectorize
    vectorizer = TfidfVectorizer()
    trainmsg = vectorizer.fit_transform(trainmsg)
    
    # actually classify and train
    my_svm = svm.SVC()
    my_svm.fit(trainmsg, trainlabel)
    
    return vectorizer, my_svm


## 6
from sklearn.metrics import confusion_matrix

trainmsg, testmsg, trainlabel, testlabel = clean_and_split_data(df, clean_func=lower_messages, test_size=0.2)

vectorizer, my_svm = train_model(trainmsg=trainmsg, 
                                 trainlabel=trainlabel)

testmsg = vectorizer.transform(testmsg)

y_pred = my_svm.predict(testmsg)
print(confusion_matrix(testlabel, y_pred))