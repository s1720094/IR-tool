import string
from nltk.stem import PorterStemmer
import re
import xml.etree.ElementTree as ET
import os
import sys
import shlex
import numpy as np
import math

# To run in command line do python code.py trec.xml queries.boolean.txt queries.ranked.txt

def preprocess(contents):
    """
    Preprocess text

    @param contents: string representing the unprocessed text
    @return: list of processed tokens
    """
    # Tokenisation
    punc = list(string.punctuation)
    punc = ''.join(punc)


    # ALternative splitting on all non alphabetic characters
    tokens = re.split("[^\w]", contents)
    if ('' in tokens):
        tokens.remove('')

    # Case folding
    # Convert all words to lower case
    lowercase_tokens = [token.lower() for token in tokens]

    # Stopping
    # Create a list of all stopwords
    with open('stopwords.txt', 'r') as f:
        stopwords_file = f.read()

    # Remove all punctuation from stopwords
    stopwords = stopwords_file.split()
    stopwords = [word.replace("'", "") for word in stopwords]
    stopped_tokens = [token for token in lowercase_tokens if token not in stopwords]

    # Normalisation
    # Initialize Porter Stemmer
    ps = PorterStemmer()

    # Perform stemming on data
    stemmed_tokens = [ps.stem(token) for token in stopped_tokens]
    return stemmed_tokens

def parse_XML(file_path):
    """
    Take in XML and parse it to token list

    @param contents: file path pointing to XML file
    @return: tuple of dictionary of documents where the key is the document id and the
             value is the preprocessed tokens for that file and the List of all doc IDs
    """
    root = ET.parse(file_path).getroot()

    # Dictionary to store each document id and its corresponding text (including the headline)
    doc_tokens = {}
    id_list = []

    # Retrieve information for each document
    for item in root.iter('document'):
        for doc in item.iter('DOC'):
            ex_text = ""
            doc_no = doc.find('DOCNO').text

            doc_headline = doc.find('HEADLINE')
            doc_text = doc.find('Text')
            doc_TEXT = doc.find('TEXT')

            # Get document headline if there is one
            if(doc_headline != None):
                ex_text = doc_headline.text + " "

            # Get document text when tag is 'Text'
            if(doc_text != None and doc_TEXT == None):
                ex_text = ex_text + doc_text.text

            # Get document text when tag is 'TEXT'
            if(doc_text == None and doc_TEXT != None):
                ex_text = ex_text + doc_TEXT.text

            # Preprocess text and add it to text dictionary
            doc_tokens[doc_no] = preprocess(ex_text)
            id_list.append(doc_no)

    return (doc_tokens, id_list)

def index(doc_tokens):
    """
    Create inverted index

    @param contents: dictionary of tokens for each document
    @return: inverted index as a dictionary where the key is the term and the
             value is a tuple where the first element is the document in which the
             term occured and the second element is a list of all positions in that
             document in which the term appeared
    """

    # These two for loops could probably be merge into one
    index = {}
    for item in doc_tokens:
        # Dictionary where key is term and value is list of positions of occurence
        # for current document
        doc_index = {}
        for i in range(0, len(doc_tokens[item])):
            token = doc_tokens[item][i]
            # Add position of occurence for term to dictionary
            if (token in doc_index):
                doc_index[token].append(i)
            else :
                doc_index[token] = [i]

        # Merge current dictionary for term to inverted index
        for elem in doc_index:
            # Add tuple of document id and list of positons of occurence
            if (elem in index):
                index[elem].append((int(item), doc_index[elem]))
            else :
                index[elem] = [(int(item), doc_index[elem])]

    # Add term frequency
    #print(index)
    return index

def print_index(index):
    """
    Pretty print into file

    @param contents: Inverted index dictionary
    """

    # Format for pretty print
    # TODO: ADD TERM FREQUENCY!!!
    out_string = ""
    for key, value in index.items():
        out_string += key + ":" + str(len(value)) + '\n'
        for doc in value:
            out_string += ('\t' + str(doc[0]) + ": " + ",".join(map(str, doc[1])) + '\n')

    with open("index.txt", "w") as outfile:
         outfile.write(out_string)

def word_search(word, index):
    """
    Find single word in index

    @param word: the word you are searching for
    @param index: the index to search through
    @return: return list of documents in which the word occurs
    """
    results = []
    if(word in index):
        results = list(map(lambda tok: tok[0], index[word]))
    return results

def phrase_search(phrase, index):
    """
    Find phrase in index

    @param phrase: the phrase you are searching for
    @param index: the index to search through
    @return: return list of documents in which the phrase occurs
    """
    word1 = phrase[0]
    word2 = phrase[1]
    results = []

    if(word1 in index and word2 in index):
        word1_docs = index[word1]
        word2_docs = index[word2]
        for tup1 in word1_docs:
            for tup2 in word2_docs:
                if(tup1[0] == tup2[0]):
                    # Add 1 to every position of the second tuple (previous word)
                    # and check to see if that list of positions and the current one
                    # have any elements in common
                    plus_prev_list = [x+1 for x in tup1[1]]
                    if len(set(tup2[1]).intersection(set(plus_prev_list))) > 0:
                        results.append(tup1[0])

    return results

def proximity_search(words, index, n):
    """
    Perform proximity search for two words in index

    @param word1: the first word in the proximity search
    @param word2: the second word
    @param index: the index to search through
    @param n: the proximity of the two words
    @return: return list of documents in which the proximity of these two words occurs
    """
    results = []

    # Both words must appear in the index for proximity search to work
    word1 = words[0]
    word2 = words[1]
    if(word1 in index and word2 in index):
        word1_docs = index[word1]
        word2_docs = index[word2]

        for tup1 in word1_docs:
            for tup2 in word2_docs:
                if(tup1[0] == tup2[0]):
                    for i in tup1[1]:
                        for j in tup2[1]:
                            if (j - i <= n and j - i > 0):
                                results.append(tup1[0])
                                break
                            elif (i > j):
                                # Skip iteration if the second word's index in the document
                                # becomes less than the first ones
                                continue
                        # Break out of both loops if you find a match in the document
                        break

    return results

def search(query, index):
    """
    Read in a query and perform it on the index

    @param query: Boolean query
    @param index: the index to search through
    @return: return list of lists where each list is query number and document relevant to that query
    """
    all_results = []
    # List to store documents retrieved
    temp = []
    if (re.search("^#(.*,.*)", query)):
        # Extract n and words through regular expression
        n = int(re.search("#(.*)\(", query).group(1))
        words = preprocess(re.search("\((.*)\)", query).group(1))
        temp += (proximity_search(words, index, n))
    else:
        if (re.search(".*\sOR\s.*", query)):
            exp1 = re.search("(.*)\sOR", query).group(1)
            exp2 = re.search("OR\s(.*)", query).group(1)
            res1 = search(exp1, index)
            res2 = search(exp2, index)
            temp += list(set(res1+ res2))
            temp.sort()
        elif (re.search(".*\sAND\s.*", query)):
            exp1 = re.search("(.*)\sAND", query).group(1)
            exp2 = re.search("AND\s(.*)", query).group(1)
            res1 = search(exp1, index)
            res2 = search(exp2, index)
            temp += [x for x in res1 if x in res2]
        elif re.search("NOT.*", query):
            exp = re.search("NOT\s(.*)", query).group(1)
            res = search(exp, index)
            temp += [x for x in id_list if x not in res]
            #FIgure out how to get max doc and make array from it
        else:
            token = preprocess(query)
            if (len(token) == 1):
                temp += word_search(token[0], index)

            if (len(token) > 1):
                temp += phrase_search(token, index)

    return temp

def queries_search(queries, index):
    """
    Read list of queries and return special format of results for each query

    @param queries: List of of lists where each list is a query
    @param index: the index to search through
    @return: return list of lists where each list is query number and document relevant to that query
    """
    all_results = []
    for bool_query in queries:
        n = re.search("^([0-9]*)", bool_query).group(1)
        query = re.search("^[0-9]*\s(.*)", bool_query).group(1)
        for result in search(query, index):
            all_results.append([str(n), str(result)])
    return all_results

def print_bool_queries(bool_queries):
    """
    Pretty print into file

    @param contents: Boolean results for queries
    """
    # Format for pretty print
    out_string = ""
    for result in bool_queries:
        out_string += (','.join(result) + '\n')

    with open("results.boolean.txt", "w") as outfile:
         outfile.write(out_string)


def tfidf(queries, index, doc_tokens):
    """
    Create inverted index

    @param queries: List of of lists where each list is a query
    @param index: The indexed documents
    @param doc_tokens: All the documents and their tokens used to calculate the
                       number of documents in the collection
    @return: The tfidf score for each query and each relevant document to the query
    """
    results = []
    N = len(doc_tokens)
    for query in queries:
        n = re.search("^([0-9]*)", query).group(1)
        words = preprocess(re.search("^[0-9]*(.*)", query).group(1))
        doc_score = {}
        for word in words:
            if (word in index):
                for tup in index[word]:
                    wtd = (1 + math.log10(len(tup[1])))*math.log10(N/len(index[word]))
                    # If score for document already contains a wtd for another word
                    # append current word wtd else add new score for current doc
                    if (tup[0] in doc_score):
                        doc_score[tup[0]] += wtd
                    else:
                        doc_score[tup[0]] = wtd

        count = 0
        for doc in sorted(doc_score, key=doc_score.get, reverse = True):
            results.append([n, str(doc), format(doc_score[doc], '.4f')]) # str(round(doc_score[doc], 4))])
            count += 1
            if(count >= 150):
                break

    return results

def print_tfidf(tfidf):
    """
    Pretty print into file

    @param contents: TFIDF results for queries
    """

    # Format for pretty print
    out_string = ""
    for result in tfidf:
        out_string += (','.join(result) + '\n')

    with open("results.ranked.txt", "w") as outfile:
         outfile.write(out_string)


# Get index from collection
sep = os.path.sep
filename = sys.argv[1]
file_path = os.getcwd() + sep + filename

parse = parse_XML(file_path)
doc_tokens = parse[0]
id_list = parse[1]
index = index(doc_tokens)
print_index(index)

# # Get boolean queries
# boolean_query_file = open(sys.argv[2], 'r', encoding="utf-8-sig")
# boolean_queries = boolean_query_file.readlines()
# boolean_queries = [q.replace('\"','') for q in boolean_queries]

# Get queries for ranked IR
ranked_query_file = open(sys.argv[2], 'r', encoding="utf-8-sig")
ranked_queries = ranked_query_file.readlines()

# print_bool_queries(queries_search(boolean_queries, index))
print_tfidf(tfidf(ranked_queries, index, doc_tokens))
