#!/bin/bash
matryoshka_vis_token_scale=$1
CKPT=llava-v1.5-7b-m3
python -m llava.eval.model_vqa_loader \
    --model-path /media/volume/new_volume/mvlm/checkpoints/llava-v1.57b-stf+m3-ori \
    --question-file /media/volume/new_volume/mvlm/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /media/volume/new_volume/mvlm/playground/data/eval/textvqa/train_images \
    --answers-file /media/volume/new_volume/mvlm/playground/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --matryoshka_vis_token_scale 576 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /media/volume/new_volume/mvlm/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /media/volume/new_volume/mvlm/playground/data/eval/textvqa/answers/${CKPT}.jsonl
