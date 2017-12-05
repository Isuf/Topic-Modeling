import numpy as np
from collections import defaultdict
import re
import pickle


############################################
# This file contains method to load the data for Machine Learning
# Author: Isuf DELIU
############################################




def documentCleansing(string):
    """ Tokenization/string cleaning """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


#############################
# Loads the data from dataPaths as:
# "content": {document_text}
# "class": {document_class}
# "num_words": {number_words}
# "split": used for Cross Validation purposes
#############################
def loadTextDocuments(dataPaths,
                      cleans_document=True, all_lowercase=True, remove_numbers=False, split_for_cv=True):
    document_records = []
    vocab = defaultdict(float)
    minWordLength = 0  # Used to Remove documents with less than 'minWordlength' words. Set -1 if not applicable
    number_of_folds = 10  # when split_for_cv == True

    for i in range(len(dataPaths)):
        with open(dataPaths[i], "r", encoding="utf-8") as fileX:
            for line in fileX:
                rev = []
                rev.append(line.strip())  # Remove Whitespace
                text = " ".join(rev)
                if cleans_document:
                    docText = documentCleansing(text)  # Document Cleansing
                if all_lowercase:
                    docText = text.lower()  # Lowercase the text
                if remove_numbers:  # To be Checked
                    docText = ' '.join(['' if word.isdigit() else word for word in text.split()])

                if (len(docText.split()) > minWordLength):
                    documentWords = set(docText.split())
                    for word in documentWords:
                        vocab[word] += 1  # Word Frequency

                    datum = {"class": i,
                             "content": docText,
                             "num_words": len(docText.split()),
                             "num_chars": len(docText),
                             "split": np.random.randint(0, number_of_folds) if (split_for_cv) else 0
                             }
                    document_records.append(datum)
    result = {"data": document_records, "vocab": vocab}
    return result


def save_topic_model( topic_model, file_path):
    pickle.dump(topic_model, open(file_path, 'wb'))


def load_topic_model(file_path):
    return pickle.load(open(file_path, 'rb'))


def top_documents_per_topic (lda, tf, data, num_topics, number_of_documents, topic_number):
        # print sample documents per topic

        doc_topic = lda.transform(tf)
        num_docs, num_topics = doc_topic.shape
        dic = {}
        for i in range(num_topics):
            dic[str(i)] = [doc_topic[doc_idx][i] for doc_idx in range(num_docs)]

        ind = np.asarray(dic[str(topic_number)]).argsort()[-number_of_documents:][::-1]
        for i in range(len(ind)):
            print(data[ind[i]])
            print()