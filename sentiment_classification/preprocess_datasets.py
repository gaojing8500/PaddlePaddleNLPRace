'''
Author: your name
Date: 2021-06-24 13:50:07
LastEditTime: 2021-06-24 19:38:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PaddlePaddleNLPRace/sentiment_classification/preprocess_data.py
'''

from six import with_metaclass
from config import Config
import pandas as pd
from sklearn.utils import shuffle
class PreProcessDataSets():
    def __init__(self):
        self.name = "数据预处理"
        self.config = Config()

    def readfiles(self,dataset_name_list,task_name,datasets):
        data_list = []
        for dataset_name in dataset_name_list:
            dataset_path = self.config.directory_structure[task_name+"-"+ datasets +"-"+ dataset_name]
            data = pd.read_csv(dataset_path,sep="\t")
            data_list.append(data)
        return data_list

    def readfiles_data_list(self,data_path_list,write_data_path):
        data_list_merge_shuffle = []
        for data_path in data_path_list:
            with open(data_path,"r",encoding="utf-8") as f:
                for line in f:
                    data_list_merge_shuffle.append(line)
            f.close()
        data_list_merge_shuffle = shuffle(data_list_merge_shuffle)
        print(len(data_list_merge_shuffle))
        self.write_data(write_data_path,data_list_merge_shuffle)

    def write_data(self,write_data_path,data_list):
        with open(write_data_path,"w",encoding='utf-8') as f:
            for line in data_list:
                f.write(line)



        
if __name__ == "__main__":
    config = Config()
    model = PreProcessDataSets()

    # data_path_list_train = [config.directory_structure["aspect_data_train_came"],config.directory_structure["aspect_data_train_phns"]]
    # data_path_list_test =  [config.directory_structure["aspect_data_test_came"],config.directory_structure["aspect_data_test_phns"]]

    # model.readfiles_data_list(data_path_list_train,config.directory_structure["aspect_data_train_merge"])
    # model.readfiles_data_list(data_path_list_test,config.directory_structure["aspect_data_test_merge"])


    # data_path_list_train = [config.directory_structure["sentences_data_train_ChnSentiCorp"],config.directory_structure["sentences_data_train_NLPCC14-SC"]]
    # data_path_list_test =  [config.directory_structure["sentences_data_test_ChnSentiCorp"],config.directory_structure["sentences_data_test_NLPCC14-SC"]]

    # model.readfiles_data_list(data_path_list_train,config.directory_structure["sentences_data_train_merge"])
    # model.readfiles_data_list(data_path_list_test,config.directory_structure["sentences_data_test_merge"])


    
    data_path_list_train = [config.directory_structure["option_data_train_bd"],
                                    config.directory_structure["option_data_train_dp"],
                                    config.directory_structure["option_data_train_mfw"]]
    data_path_list_test = [config.directory_structure["option_data_test_bd"],
                                    config.directory_structure["option_data_test_dp"],
                                    config.directory_structure["option_data_test_mfw"]]

    model.readfiles_data_list(data_path_list_train,config.directory_structure["option_data_train_merge"])
    model.readfiles_data_list(data_path_list_test,config.directory_structure["option_data_test_merge"])

    # data_list = model.readfiles(["SE-ABSA16_CAME","SE-ABSA16_PHNS"],"aspect","test")
    # data_merge = data_list[0].append(data_list[1])
    # print(len(data_merge))
    # data_merge_shuffle = shuffle(data_merge)
    # data_merge.to_csv(config.directory_structure["aspect_data_test_merge"])


    # data_list = model.readfiles(["COTE-BD","COTE-DP","COTE-MFW"],"option","train")
    # data_merge = data_list[0].append(data_list[1].append(data_list[2]))
    # print(len(data_merge))
    # data_merge_shuffle = shuffle(data_merge)
    # data_merge.to_csv(config.directory_structure["option_data_train_merge"])


    # data_list = model.readfiles(["ChnSentiCorp","NLPCC14-SC"],"sentences","train")
    # data_merge = data_list[0].append(data_list[1].append(data_list[2]))
    # print(len(data_merge))
    # data_merge_shuffle = shuffle(data_merge)
    # data_merge.to_csv(config.directory_structure["sentences_data_train_merge"])



    


    
        


