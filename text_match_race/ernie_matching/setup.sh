###
 # @Author: your name
 # @Date: 2021-06-07 21:09:19
 # @LastEditTime: 2021-06-09 14:24:52
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /PTM/PaddleNLP-develop/examples/text_matching/ernie_matching/setup.sh
### 
$ unset CUDA_VISIBLE_DEVICES
ROOT_PATH_MODEL=/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/model/checkpoints
python -u -m paddle.distributed.launch --gpus "0,1,2" train_pointwise.py \
        --device gpu \
        --save_dir ${ROOT_PATH_MODEL}/V1.0 \
        --batch_size 32 \
        --learning_rate 2E-5