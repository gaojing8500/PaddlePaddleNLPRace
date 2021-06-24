###
 # @Author: your name
 # @Date: 2021-06-24 16:12:15
 # @LastEditTime: 2021-06-24 19:55:01
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /PaddlePaddleNLPRace/sentiment_classification/task-aspect/train.sh
### 

$ unset CUDA_VISIBLE_DEVICES
ROOT_PATH_MODEL=/home/gaojing/PTM/datasets/qian_yan_text_sentiment_analysis/model/opinion_model/checkpoints
python -u -m paddle.distributed.launch --gpus "0" train_opinion.py \
        --device gpu \
        --save_dir ${ROOT_PATH_MODEL}/baseline \
        --batch_size 2 \
        --learning_rate 3e-6\
        --epochs 1