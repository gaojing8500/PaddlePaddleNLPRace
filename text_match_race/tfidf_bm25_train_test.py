'''
Author: your name
Date: 2021-06-07 10:59:42
LastEditTime: 2021-06-07 20:31:29
LastEditors: Please set LastEditors
Description: bm25和tfidf算法测试和训练
FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/qian_yan_text_match/tfidf_bm25_train_test.py
'''


from tfidf_bm25_text_match import BM25
from tfidf_bm25_text_match import TFIDF
from tokenizer import JiebaTokenizer
from config import Config
import jieba
class TestBm25Tfidf(object):
    def __init__(self,test_data_path):
        self.test_path = test_data_path
        self.document_list = []
        self.q_text_list = []
        self.a_text_list = []
        self.read_file()
        self.q_text_list_tokenizer()
    
    def read_file(self):
        with open(self.test_path,'r',encoding = 'utf-8') as f:
            for line in f:
                q_text,a_text,label = line.strip().split("\t")
                self.q_text_list.append(q_text)
                self.a_text_list.append(a_text)
    def q_text_list_tokenizer(self):
        tokenizer = JiebaTokenizer()
        for sent in self.q_text_list:
            sentence_token_list = []
            ##jieba分词
            sentence_token = tokenizer.tokenize(sent)
            for token in sentence_token:
                sentence_token_list.append(token[0])
            self.document_list.append(sentence_token_list)
    def test_bm25_model(self,query):
        bm25_model = BM25(self.document_list)
        query_list = list(jieba.cut(query))
        index_score = bm25_model.get_documents_score(query_list)
        ##top-5相似问题
        q_similarity = []
        score = []
        index_score = index_score[0:5]
        for i in index_score:
            q_similarity.append(self.q_text_list[i['index']])
            score.append(i['score'])
        print("bm25model_相似问题：{}".format(q_similarity))
        print("bm25model_相似分数：{}".format(score))
    def test_tfidf_model(self,query):
        tfidf_model = TFIDF(self.document_list)
        query_list = list(jieba.cut(query))
        index_score = tfidf_model.get_documents_score(query_list)
        ##top-5相似问题
        q_similarity = []
        score = []
        index_score = index_score[0:5]
        for i in index_score:
            q_similarity.append(self.q_text_list[i['index']])
            score.append(i['score'])
        print("tfidf_相似问题：{}".format(q_similarity))
        print("tfidf_相似分数：{}".format(score))
    def test_gensim_tfidf(self,query,top_k):
        gensim_tfidf_model =  TFIDF(self.document_list)
        query_list = list(jieba.cut(query))
        gensim_tfidf_model.gensim_tfidf_model()
        index,score = gensim_tfidf_model.gensim_get_similarity_score(query_list,top_k)
        q_similarity = []
        score_list = []
        for index_,score_ in zip(index,score) :
            q_similarity.append(self.q_text_list[index_])
            score_list.append(score_)
        print("gensim_tfidf_相似问题：{}".format(q_similarity))
        print("gensim_tfidf_相似分数：{}".format(score))


if __name__ == '__main__':
    config = Config()
    test_data_path = config.directory_structure["merge_train_datasets"]
    test_bm25_tfidf = TestBm25Tfidf(test_data_path)
    while(1):
        user_query = input("请输入您的问题: \n")
        if user_query == 'quit':
            break
        else:
            # test_bm25_tfidf.test_bm25_model(user_query)
            # test_bm25_tfidf.test_tfidf_model(user_query)
            test_bm25_tfidf.test_gensim_tfidf(user_query,5)