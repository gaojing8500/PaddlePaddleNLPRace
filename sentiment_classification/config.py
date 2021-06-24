'''
Author: your name
Date: 2021-06-24 13:50:48
LastEditTime: 2021-06-24 16:58:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PaddlePaddleNLPRace/sentiment_classification/config.py
'''

class Config():
    def __init__(self):
        self.name ="配置数据路径"
        self.root_path = "/home/gaojing/PTM/datasets/qian_yan_text_sentiment_analysis/"
        self.task_name = ["aspect","option","sentences"]
        self.datasets_name_aspect = ["SE-ABSA16_CAME","SE-ABSA16_PHNS"]
        self.datasets_name_option = ["COTE-BD","COTE-DP","COTE-MFW"]
        self.datasets_name_sentences = ["ChnSentiCorp","NLPCC14-SC"]
        self.datasets_name_dict_write_path = {}
        self.directory_structure = {
            "aspect_data_train_merge":self.root_path + self.task_name[0] + "/" + "train.tsv",
            "aspect_data_test_merge":self.root_path + self.task_name[0] + "/" + "test.tsv",

            "option_data_train_merge":self.root_path + self.task_name[1] + "/" + "train.tsv",
            "option_data_test_merge":self.root_path + self.task_name[1] + "/" + "test.tsv",

            "sentences_data_train_merge":self.root_path + self.task_name[2] + "/" + "train.tsv",
            "sentences_data_test_merge":self.root_path + self.task_name[2] + "/" + "test.tsv",

            "aspect_data_train_came":self.root_path + self.task_name[0] +"/"+self.datasets_name_aspect[0] + "/" + "train.tsv",
            "aspect_data_test_came":self.root_path + self.task_name[0] +"/"+self.datasets_name_aspect[0] + "/" + "test.tsv",

            "aspect_data_train_phns":self.root_path + self.task_name[0] +"/"+self.datasets_name_aspect[1] + "/" + "train.tsv",
            "aspect_data_test_phns":self.root_path + self.task_name[0] +"/"+self.datasets_name_aspect[1] + "/" + "test.tsv",

            
            "option_data_train_bd":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[0] + "/" + "train.tsv",
            "option_data_test_bd":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[0] + "/" + "test.tsv",

            "option_data_train_dp":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[1] + "/" + "train.tsv",
            "option_data_test_dp":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[1] + "/" + "test.tsv",

            "option_data_train_mfw":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[2] + "/" + "train.tsv",
            "option_data_test_mfw":self.root_path + self.task_name[1] +"/"+self.datasets_name_option[2] + "/" + "test.tsv",

            "sentences_data_train_ChnSentiCorp":self.root_path + self.task_name[2] +"/"+self.datasets_name_sentences[0] + "/" + "train.tsv",
            "sentences_data_test_ChnSentiCorp":self.root_path + self.task_name[2] +"/"+self.datasets_name_sentences[0] + "/" + "test.tsv",
            
        
            "sentences_data_train_NLPCC14-SC":self.root_path + self.task_name[2] +"/"+self.datasets_name_sentences[1] + "/" + "train.tsv",
            "sentences_data_test_NLPCC14-SC":self.root_path + self.task_name[2] +"/"+self.datasets_name_sentences[1] + "/" + "test.tsv",
        
        }
        # self.init_data_directory_structure()

    def __str__(self):
        return self.name
    def inset_data_map(self,key,task_name,datasets_name,value):
        self.directory_structure[key] = self.root_path + task_name + "/" + datasets_name + "/" + value
    def init_data_directory_structure(self):
        for index_task in self.task_name:
            if index_task == "aspect":
                for datasets_index in self.datasets_name_aspect:
                    self.inset_data_map("aspect-train"+"-"+datasets_index,index_task,datasets_index,"train.tsv")
                    self.inset_data_map("aspect-test"+"-"+datasets_index,index_task,datasets_index,"test.tsv")
            if index_task == "option":
                for datasets_index in self.datasets_name_option:
                    self.inset_data_map("option-train"+"-"+datasets_index,index_task,datasets_index,"train.tsv")
                    self.inset_data_map("option-test"+"-"+datasets_index,index_task,datasets_index,"test.tsv")
            if index_task == "sentences":
                for datasets_index in self.datasets_name_option:
                    self.inset_data_map("sentences-train"+"-"+datasets_index,index_task,datasets_index,"train.tsv")
                    self.inset_data_map("sentences-test"+"-"+datasets_index,index_task,datasets_index,"test.tsv")

if __name__ == "__main__":
    config = Config()
    print(config.directory_structure["aspect-train-SE-ABSA16_CAME"])

        

    


    