#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3
python -m llava.eval.model_vqa_loader \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /media/volume/new_volume/mvlm/playground/data/eval/pope/val2014 \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/pope/answers/$CKPT.jsonl \
    --matryoshka_vis_token_scale 576 \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /media/volume/new_volume/mvlm/playground/data/eval/pope/coco \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /media/volume/new_volume/mvlm/playground/data/eval/pope/answers/$CKPT.jsonl
