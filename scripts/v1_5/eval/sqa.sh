#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3-${matryoshka_vis_token_scale}
python -m llava.eval.model_vqa_science \
    --model-path /media/volume/pan_2000G/mvlm/checkpoints/llava-v1.57b-stf+m3 \
    --question-file /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/images/test \
    --answers-file /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa \
    --result-file /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /media/volume/pan_2000G/mvlm/playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
