###
 # @Author: your name
 # @Date: 2021-06-07 22:15:06
 # @LastEditTime: 2021-06-10 15:18:46
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /PTM/PaddleNLP-develop/examples/text_matching/ernie_matching/setup_predict.sh
### 
$ unset CUDA_VISIBLE_DEVICES
ROOT_PATH_MODEL=/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/model/checkpoints/merge_checkpoints_epoch_5
python -u -m paddle.distributed.launch --gpus "1" \
        predict_pointwise.py \
        --device gpu \
        --params_path ${ROOT_PATH_MODEL}/model_15700/model_state.pdparams \
        --batch_size 128 \
        --max_seq_length 128 \
        --input_file '/home/gaojing/PTM/datasets/qian_yan_text_match_datasets/paws-x-zh/test.tsv'