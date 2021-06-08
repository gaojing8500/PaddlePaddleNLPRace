'''
Author: your name
Date: 2021-06-07 10:34:26
LastEditTime: 2021-06-07 20:35:44
LastEditors: Please set LastEditors
Description: 传统文本匹配算法 BM25 TDIDF算法 
FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/qian_yan_text_match/tfidf_bm25_text_match.py
'''

import numpy as np
from collections import defaultdict
from collections import Counter
from gensim import corpora,models,similarities
class BM25(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.name = "bm25算法"
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 文本库中文本的平均长度
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        # 存储每个文本中每个词的词频
        self.f = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            ##加权分数作为相似度评价
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
        return score

    def get_documents_score(self, query):
        temp_score_list = []
        for i in range(self.documents_number):
            score_dict = {}
            #score_list.append(self.get_score(i, query))
            score_dict['index'] = i
            score_dict['score'] = self.get_score(i, query)
            temp_score_list.append(score_dict)
        ##重新排序
        score_sorted = sorted(temp_score_list,key = lambda e:e.__getitem__('score'),reverse = True)
        return score_sorted


class TFIDF(object):

    def __init__(self, documents_list):
        # 文本列表，内部每个文本需要事先分好词
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 存储每个文本中每个词的词频
        self.tf = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            ##加权分数作为相似度评分,并不是将句子向量化，再计算相似度(余弦距离)
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_documents_score(self, query):
        temp_score_list = []
        for i in range(self.documents_number):
            score_dict = {}
            #score_list.append(self.get_score(i, query))
            score_dict['index'] = i
            score_dict['score'] = self.get_score(i, query)
            temp_score_list.append(score_dict)
        ##重新排序
        score_sorted = sorted(temp_score_list,key = lambda e:e.__getitem__('score'),reverse = True)
        return score_sorted
    ##对句子进行tdidf向量化，在计算句子之间相似度
    def gensim_corpus_model(self):
        #texts = []
        import pdb
        
        min_frequency = 1
        frequency = defaultdict(int)
        for sentence in self.documents_list:
            for token in sentence:
                frequency[token] += 1
        pdb.set_trace()
        self.texts = [[token for token in text if frequency[token] > min_frequency] for text in self.documents_list]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def gensim_tfidf_model(self):
        self.gensim_corpus_model()
        self.tfidf_model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.tfidf_model[self.corpus_simple]
        ##向量化矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    def gensim_get_similarity_score(self,query,top_k):
        vec_bow = self.dictionary.doc2bow(query)
        sentence_vec = self.tfidf_model[vec_bow]
        similarity_sentence = self.index[sentence_vec]
        sim_top_k = sorted(enumerate(similarity_sentence), key=lambda item: item[1], reverse=True)[:top_k]
        indexs = [i[0] for i in sim_top_k]
        scores = [i[1] for i in sim_top_k]
        return indexs,scores

       
