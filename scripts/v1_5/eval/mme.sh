#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3

python -m llava.eval.model_vqa_loader \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /media/volume/new_volume/mvlm/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

cd /media/volume/new_volume/mvlm/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
