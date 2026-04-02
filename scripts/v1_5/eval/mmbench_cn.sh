#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3
SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

mkdir -p /media/volume/new_volume/mvlm/playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /media/volume/new_volume/mvlm/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /media/volume/new_volume/mvlm/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /media/volume/new_volume/mvlm/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
