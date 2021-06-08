'''
Author: your name
Date: 2021-06-07 10:51:41
LastEditTime: 2021-06-08 10:21:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/qian_yan_text_match/config.py
'''
class Config():
    def __init__(self):
        self.name = "配置数据集路径"
        self.root_path = "/home/gaojing/PTM/datasets/qian_yan_text_match_datasets"
        self.merge_train_datasets = "/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/merge_datasets/"
        self.directory_structure = {
            "bq_corpus_train": self.root_path +"/bq_corpus/train.tsv",
            "bq_corpus_test": self.root_path +"/bq_corpus/test.tsv",
            "bq_corpus_dev": self.root_path +"/bq_corpus/dev.tsv",
            "lcqmc_train": self.root_path +"/lcqmc/train.tsv",
            "lcqmc_test": self.root_path +"/lcqmc/test.tsv",
            "lcqmc_dev": self.root_path +"/lcqmc/dev.tsv",
            "paws-x-zh_train": self.root_path +"/paws-x-zh/train.tsv",
            "paws-x-zh_test": self.root_path +"/paws-x-zh/test.tsv",
            "paws-x-zh_dev": self.root_path +"/paws-x-zh/dev.tsv",
            "merge_train_datasets_train": self.merge_train_datasets+ "merge_bq_lc_paws_datasets_train_shuffle.tsv",
            "merge_dev_datasets_dev": self.merge_train_datasets+ "merge_bq_lc_paws_datasets_dev_shuffle.tsv",
            "merge_test_datasets_test": self.merge_train_datasets+ "merge_bq_lc_paws_datasets_test_shuffle.tsv"
        }
        

        



