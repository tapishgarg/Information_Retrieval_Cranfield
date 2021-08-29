# Add your import statements here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from evaluation import Evaluation

# Add any utility functions here

def Evaluation_metrics(doc_IDs_ordered, query_ids, qrels, title_name = " "):
    evaluator = Evaluation()
    precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
    for k in range(1,11):
        precision = evaluator.meanPrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        precisions.append(precision)
        recall = evaluator.meanRecall(
            doc_IDs_ordered, query_ids, qrels, k)
        recalls.append(recall)
        fscore = evaluator.meanFscore(
            doc_IDs_ordered, query_ids, qrels, k)
        fscores.append(fscore)

        MAP = evaluator.meanAveragePrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        MAPs.append(MAP)
        nDCG = evaluator.meanNDCG(
            doc_IDs_ordered, query_ids, qrels, k)
        nDCGs.append(nDCG)
        
        print("Precision, Recall and F-score @ " +  
            str(k) + " : " + str(precision) + ", " + str(recall) + 
            ", " + str(fscore))
        print("MAP, nDCG @ " +  
            str(k) + " : " + str(MAP) + ", " + str(nDCG))

    plt.figure(figsize=(10,5))
    plt.plot(range(1, 11), precisions, label="Precision")
    plt.plot(range(1, 11), recalls, label="Recall")
    plt.plot(range(1, 11), fscores, label="F-Score")
    plt.plot(range(1, 11), MAPs, label="MAP")
    plt.plot(range(1, 11), nDCGs, label="nDCG")
    plt.legend()
    plt.title(title_name)
    plt.xlabel("k")
    return

def Query_metrics(qrels, doc_IDs_ordered, queries, model_name = ' '):

    df = pd.DataFrame(qrels)
    evaluator = Evaluation()

    # Precision of each queries
    q_precision = []

    # Recall of each queries
    q_recall = []

    # Fscore of each queries
    q_fscore = []
    
    # Average Precision of each queries
    q_average_precision = []

    for i in range(len(doc_IDs_ordered)):
        true_doc_ids = list(map(int, df[df['query_num'] == str(i+1)]['id'].tolist()))
        
        precision = evaluator.queryPrecision(doc_IDs_ordered[i], i+1, true_doc_ids, 10)
        q_precision.append(precision)
        
        recall = evaluator.queryRecall(doc_IDs_ordered[i], i+1, true_doc_ids, 10)
        q_recall.append(recall)
        
        fscore = evaluator.queryFscore(doc_IDs_ordered[i], i+1, true_doc_ids, 10)
        q_fscore.append(fscore)
        
        average_precision = evaluator.queryAveragePrecision(doc_IDs_ordered[i], i+1, true_doc_ids, 10)
        q_average_precision.append(average_precision)


    # ndcg for each query calculation
    q_ndcg = []
    for i in range(len(doc_IDs_ordered)):
        true_doc_ndcg = df[df['query_num'] == str(i+1)][['position', 'id']]
        ndcg = evaluator.queryNDCG(doc_IDs_ordered[i], i+1, true_doc_ndcg, 10)
        q_ndcg.append(list(ndcg)[0])
    
    return q_precision, q_recall, q_fscore, q_ndcg, q_average_precision
 