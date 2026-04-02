#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /media/volume/pan_2000G/mvlm/checkpoints/llava-v1.57b-stf+m3 \
    --question-file /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

mkdir -p /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /media/volume/pan_2000G/mvlm/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
