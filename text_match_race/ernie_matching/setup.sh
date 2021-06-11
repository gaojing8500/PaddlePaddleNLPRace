###
 # @Author: your name
 # @Date: 2021-06-07 21:09:19
 # @LastEditTime: 2021-06-10 10:24:23
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /PTM/PaddleNLP-develop/examples/text_matching/ernie_matching/setup.sh
### 
$ unset CUDA_VISIBLE_DEVICES
ROOT_PATH_MODEL=/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/model/checkpoints
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_pointwise.py \
        --device gpu \
        --save_dir ${ROOT_PATH_MODEL}/merge_checkpoints_epoch_5 \
        --batch_size 32 \
        --learning_rate 2E-5\
        --epochs 6