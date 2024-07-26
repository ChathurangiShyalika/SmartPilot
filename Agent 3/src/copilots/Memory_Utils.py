import requests
import json
import torch
import time
import nltk
import re
import torch.nn as nn
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from bs4 import BeautifulSoup
from tqdm import tqdm
from decouple import config
from groq import Groq
from random import choice
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('wordnet')

class Cluster_Model:
    
    def __init__(self,max_depth=4):
        self.max_depth = max_depth
        self.index = {}
        self.clusters = {}
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def find_closest_and_avg(self):

        min_sim, closest_pair = 1.0, (0,0)
        min_i, min_j = None, None
        for frozen_vector_i in self.index:
            v_i = torch.tensor(list(frozen_vector_i))
            v_i_idx = self.index[frozen_vector_i]
            for frozen_vector_j in self.index:
                v_j = torch.tensor(list(frozen_vector_j))
                v_j_idx = self.index[frozen_vector_j]
                if v_i_idx == v_j_idx:
                    continue
                sim_i_j = self.cos(v_i,v_j)
                if sim_i_j <= min_sim:
                    min_sim = sim_i_j
                    closest_pair = (v_i_idx,v_j_idx)
                    min_i, min_j = v_i, v_j

        vector_pair = torch.stack([min_i,min_j])
        mean_repr = torch.mean(vector_pair,dim=0)
        return mean_repr, (closest_pair,min_sim)

    def cluster(self,demo_text_split_vectors,cut_threshold=0.0):
        n_vectors = len(demo_text_split_vectors)
        splits = [str(item)+';' for item in range(n_vectors)]
        for i in range(n_vectors):
            vector = demo_text_split_vectors[i]
            self.index[frozenset(vector.tolist())] = i

        level = 0
        while True:

            try:

                if level == self.max_depth-1:
                    break

                mean_repr, closest_pair = self.find_closest_and_avg()
                closest_pair_threshold = closest_pair[1]
                if closest_pair_threshold <= cut_threshold:
                    break
                self.index[frozenset(mean_repr.tolist())] = closest_pair[0]
                for frozen_set in list(self.index.keys()):
                    if self.index[frozen_set] in closest_pair[0]:
                        del self.index[frozen_set]
                level += 1

            except:
                break

        clusters = []

        for frozen_set in self.index:
            item, new_item = self.index[frozen_set], []
            if not type(item) == int:
                new_item += [[int(l) for l in list(re.sub(r'[^0-9]+', '', str(sub_item)))] for sub_item in item]
            else:
                new_item += [[item]]
            split_items = [''.join([splits[sub_sub_item] for sub_sub_item in sub_item]) for sub_item in new_item]
            clusters += split_items

        return clusters

    def prune_splits(self,query,text_splits,top_k=3):
        
        neural_net = Neural_Net()
        query_vector = neural_net.vectorize(query)
        query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [neural_net.vectorize(split) for split in text_splits]
        similarities = [neural_net.vector_similarity(x[0],x[1]).item() for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return '\n ===== \n'.join([text_splits[idx] for idx in top_3_idxs])

class Text_Preprocessor:
    
    @staticmethod
    def text_splitter(text, split_size=4):
        """
        splits text into splits of specified size
        """
        a, n = text, split_size
        k, m = divmod(len(a), n)
        return_list = list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        processed_return_list = []
        for item in return_list:
            processed_return_list.append(';'.join([sub_item for sub_item in item.split('\n') if item.strip()]))

        return processed_return_list

class Symbolic_Model:

    def __init__(self):

        def vector_similarity(keywords1,keywords2):
            """
            implements jaccard similarity based set similarity
            """

            A, B = set(keywords1), set(keywords2)
            # Intersaction and Union of two sets can also be done using & and | operators.
            C = A.intersection(B)
            D = A.union(B)
            return float(len(C))/float(len(D))

        def vectorize(sentence):
            """
            implements a symbolic vectorizer instance
            """

            return word_tokenize(sentence)

        self.vectorize = vectorize
        self.vector_similarity = vector_similarity

class Neural_Net:

    def __init__(self):

        def vector_similarity(vector1,vector2):
            """
            implements a vector similarity instance
            """

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            return cos(vector1,vector2)

        def vectorize(sentence):
            """
            implements a vectorizer instance
            """

            embedding_model = TransformerDocumentEmbeddings('bert-base-uncased')
            sentence = Sentence(sentence)
            embedding_model.embed(sentence)
            return sentence.embedding

        self.vectorize = vectorize
        self.vector_similarity = vector_similarity

class Retr:

    @staticmethod
    def retrieve_context_neural(text_splits,random_question,neural_net,top_k = 3):
        """
        retrieves top k context based on vector similarity search
        """

        query_vector = neural_net.vectorize(random_question); query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [neural_net.vectorize(split) for split in text_splits]
        similarities = [neural_net.vector_similarity(x[0],x[1]).item() for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return [text_splits[idx] for idx in top_3_idxs]

    def retrieve_context_symbolic(text_splits,random_question,symb_model,top_k = 3):
        """
        retrieves top k context based on symbolic search
        """
        n = len(text_splits)
        query_vector = symb_model.vectorize(random_question); query_vectors = [query_vector for _ in range(n)]
        split_vectors = [symb_model.vectorize(split) for split in text_splits]
        similarities = [symb_model.vector_similarity(x[0],x[1]) for x in zip(query_vectors,split_vectors)]
        top_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return [text_splits[idx] for idx in top_idxs]

    def retrieve_context(text_splits,query,neural_net=None,symb_model=None,top_k = 3):
        """
        retrieves top k context based on hybrid search
        """
        neural_context, symbolic_context = [], []
        if not neural_net is None:
            neural_context += Retr.retrieve_context_neural(text_splits,query,neural_net,top_k=top_k)
        if not symb_model is None:
            symbolic_context += Retr.retrieve_context_symbolic(text_splits,query,symb_model,top_k=top_k)
        return neural_context + symbolic_context

class Knowledge_Representation:

    @staticmethod
    def organize_data(article_text):
        
        cluster_obj = Cluster_Model() #levels default = 4
        text_splits = Text_Preprocessor.text_splitter(article_text,split_size=100)
        '''
        print (max([len(item) for item in text_splits])); input()
        text_split_vectors = [Neural_Net().vectorize(split) for split in tqdm(text_splits)]
        clusters = cluster_obj.cluster(text_split_vectors)
        text_clusters = []
        for cluster in clusters:
            idxs = [int(item) for item in cluster.split(';')[:-1]]
            text_cluster = ''.join([text_splits[idx] for idx in idxs])
            text_clusters.append(text_cluster)
        '''
        text_clusters = text_splits
        return text_clusters