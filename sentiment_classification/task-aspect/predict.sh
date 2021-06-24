###
 # @Author: your name
 # @Date: 2021-06-24 16:12:15
 # @LastEditTime: 2021-06-24 20:42:23
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /PaddlePaddleNLPRace/sentiment_classification/task-aspect/train.sh
### 

$ unset CUDA_VISIBLE_DEVICES
ROOT_PATH_MODEL=/home/gaojing/PTM/datasets/qian_yan_text_sentiment_analysis/model/aspect_model/checkpoints
python -u -m paddle.distributed.launch --gpus "1,2,3" predict_aspect.py \
        --params_path ${ROOT_PATH_MODEL}/baseline/model_2400/model_state.pdparams \
        --device gpu \
        --max_seq_length 500\
        --batch_size 2 