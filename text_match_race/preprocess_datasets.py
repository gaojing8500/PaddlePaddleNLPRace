'''
Author: your name
Date: 2021-06-07 10:52:07
LastEditTime: 2021-06-09 11:29:56
LastEditors: Please set LastEditors
Description: 合并三个数据集的的tran.tsv 和test.tsv
FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/qian_yan_text_match/preprocess_datasets.py
'''
import pandas as pd
from config import Config
from sklearn.utils import shuffle

class PreProcessDataSets():
    def __init__(self):
        self.name = "预处理千言数据集"
        self.config = Config()

    def readfiles(self):
        """[summary]读入文件 合并train数据集 打散输出
        """        
        train_bq_dataset = self.config.directory_structure["bq_corpus_train"]
        train_lc_dataset = self.config.directory_structure["lcqmc_train"]
        train_paw_dataset = self.config.directory_structure["paws-x-zh_train"]
        train_bq_data = pd.read_csv(train_bq_dataset,sep="  ")
        train_lc_data = pd.read_csv(train_lc_dataset,sep="\t")
        train_paws_data = pd.read_csv(train_paw_dataset,sep="\t")
        print("bq数据总条数：{}".format(len(train_bq_data)))
        print("lc数据总条数：{}".format(len(train_lc_data)))
        print("paws数据总条数：{}".format(len(train_paws_data)))
        merge_bq_lc_paws_datasets = train_bq_data.append(train_lc_data).append(train_paws_data)
        
        print("bq_lc合并数据总条数：{}".format(len(merge_bq_lc_paws_datasets)))
        merge_bq_lc_paws_datasets.to_csv(self.config.merge_train_datasets + "merge_bq_lc_paws_datasets_train.tsv",sep="\t")
        # 数据打散
        merge_bq_lc_paws_datasets_shuffle = shuffle(merge_bq_lc_paws_datasets)
        merge_bq_lc_paws_datasets_shuffle.to_csv(self.config.merge_train_datasets + "merge_bq_lc_paws_datasets_train_shuffle.tsv",sep="\t")

    def readfiles_refacting(self):
        merge_bq_lc_paws_datasets =[]
        train_bq_dataset = self.config.directory_structure["bq_corpus_train"]
        train_lc_dataset = self.config.directory_structure["lcqmc_train"]
        train_paw_dataset = self.config.directory_structure["paws-x-zh_train"]
        with open(train_bq_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                    merge_bq_lc_paws_datasets.append(line)
        with open(train_lc_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        with open(train_paw_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        print("bq_lc合并数据总条数：{}".format(len(merge_bq_lc_paws_datasets)))
        merge_bq_lc_paws_datasets_shuffle = shuffle(merge_bq_lc_paws_datasets)
        with open(self.config.merge_train_datasets + "merge_bq_lc_paws_datasets_train_shuffle.tsv",'w',encoding="utf-8") as f:
            for line in merge_bq_lc_paws_datasets_shuffle:
                f.write(line)

    def readfiles_refacting_dev(self):
        merge_bq_lc_paws_datasets =[]
        train_bq_dataset = self.config.directory_structure["bq_corpus_dev"]
        train_lc_dataset = self.config.directory_structure["lcqmc_dev"]
        train_paw_dataset = self.config.directory_structure["paws-x-zh_dev"]
        with open(train_bq_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                    merge_bq_lc_paws_datasets.append(line)
        with open(train_lc_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        with open(train_paw_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        print("bq_lc合并数据总条数：{}".format(len(merge_bq_lc_paws_datasets)))
        merge_bq_lc_paws_datasets_shuffle = shuffle(merge_bq_lc_paws_datasets)
        with open(self.config.merge_train_datasets + "merge_bq_lc_paws_datasets_dev_shuffle.tsv",'w',encoding="utf-8") as f:
            for line in merge_bq_lc_paws_datasets_shuffle:
                f.write(line)

    def readfiles_refacting_test(self):
        merge_bq_lc_paws_datasets =[]
        train_bq_dataset = self.config.directory_structure["bq_corpus_test"]
        train_lc_dataset = self.config.directory_structure["lcqmc_test"]
        train_paw_dataset = self.config.directory_structure["paws-x-zh_test"]
        with open(train_bq_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                    merge_bq_lc_paws_datasets.append(line)
        with open(train_lc_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        with open(train_paw_dataset,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                        merge_bq_lc_paws_datasets.append(line)
        print("bq_lc合并数据总条数：{}".format(len(merge_bq_lc_paws_datasets)))
        merge_bq_lc_paws_datasets_shuffle = shuffle(merge_bq_lc_paws_datasets)
        with open(self.config.merge_train_datasets + "merge_bq_lc_paws_datasets_test_shuffle.tsv",'w',encoding="utf-8") as f:
            for line in merge_bq_lc_paws_datasets_shuffle:
                f.write(line)

    def readfiles(self,file_path):
        data_list = []
        with open(file_path,'r',encoding="utf-8") as f:
            for line in f:
                if(len(line.strip().split('\t')) != 1):
                    data_list.append(line)
        return data_list

    def writefiles(self,files_out_path,data_list):
        with open(files_out_path,'w',encoding="utf-8") as f:
            for line in data_list:
                f.write(line)

    def paws_x_zh_preprocess(self):
        train_paw_dataset_train = self.config.directory_structure["paws-x-zh_train"]
        train_paw_dataset_dev = self.config.directory_structure["paws-x-zh_dev"]
        train_paw_dataset_test = self.config.directory_structure["paws-x-zh_test"]
        root_path = "/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/paws-x-zh/preprocess_data/"
        train = self.readfiles(train_paw_dataset_train)
        dev = self.readfiles(train_paw_dataset_dev)
        test = self.readfiles(train_paw_dataset_test)
        self.writefiles(root_path + "train.tsv",train)
        self.writefiles(root_path + "dev.tsv",dev)
        self.writefiles(root_path + "test.tsv",test)


        


    def test_readfiles(self):
        test_data_path = self.config.directory_structure["merge_train_datasets"]
        with open(test_data_path,"r",encoding ="utf-8") as f:
            for line in f:
                if(len(line.strip().split("\t")) != 1):
                    q_text,a_text,label = line.strip().split("\t")
                



    def writfiles(self,merge_bq_lc_paws_datasets):
        return "合并写入具体的路径"

    def hanlp_tokenize(self,mod):
        """[summary]采用 hanlp进行分词处理

        Args:
            mod ([type]): [description] 选择hanlp 分词模式
        """        


if __name__ =="__main__":
    preProcessDataSets = PreProcessDataSets()
    # preProcessDataSets.test_readfiles()
    # preProcessDataSets.readfiles_refacting_dev()
    # preProcessDataSets.readfiles_refacting_test()
    preProcessDataSets.paws_x_zh_preprocess()


