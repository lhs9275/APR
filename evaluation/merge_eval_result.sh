#!/bin/sh

#================ defects4j =======================
# 1. codellama-7b
python3 merge_eval_result.py \
    --dataset defects4j \
    --model_inference_dirs "codellama_7b_instruct_fp16_Instruction" \
    --history_settings "1,2,3,4,5,6,7,8" \

# 2. deepseek-coder
python3 merge_eval_result.py \
    --dataset defects4j \
    --model_inference_dirs "deepseek_coder_6.7b_instruct_fp16_Instruction,deepseek_coder_v2_16b_lite_instruct_fp16_Instruction" \
    --history_settings "1,2,3,4,5,6,7,8" \


#================ bugsinpy =======================
python3 merge_eval_result.py \
    --dataset bugsinpy \
    --model_inference_dirs "codellama_7b_instruct_fp16_Instruction,codellama_7b_instruct_fp16_InstructionLabel,codellama_7b_instruct_fp16_InstructionMask" \
    --history_settings "1,2,3,4,5,6,7,8" \


## baseline 60
#python3 merge_eval_result.py \
#    --dataset defects4j \
#    --model_inference_dirs "codellama_7b_instruct_fp16_InstructionLabel,codellama_7b_instruct_fp16_InstructionMask,deepseek_coder_6.7b_instruct_fp16_InstructionLabel,deepseek_coder_6.7b_instruct_fp16_InstructionMask,deepseek_coder_v2_16b_lite_instruct_fp16_InstructionLabel,deepseek_coder_v2_16b_lite_instruct_fp16_InstructionMask" \
#    --history_settings "1,2,3,4,5,6,7,8" \

#python3 merge_eval_result.py \
#    --dataset defects4j \
#    --model_inference_dirs "codellama_7b_instruct_fp16_Instruction,deepseek_coder_6.7b_instruct_fp16_Instruction,deepseek_coder_v2_16b_lite_instruct_fp16_Instruction" \
#    --history_settings "1" \
