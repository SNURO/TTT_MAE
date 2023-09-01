DATA_PATH_BASE='/gallery_tate/wonjae.roh/imagenetC'
DATASET='gaussian_noise'
LEVEL='5'
RESUME_MODEL='checkpoints/mae_pretrain_vit_large_full.pth'
RESUME_FINETUNE='checkpoints/prob_lr1e-3_wd.2_blk12_ep20.pth'
OUTPUT_DIR_BASE='outputs'

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=2222

python main_test_time_training.py \
    --data_path "$DATA_PATH_BASE/$DATASET/$LEVEL" \
    --model mae_vit_large_patch16 \
    --input_size 224 \
    --batch_size 32 \
    --steps_per_example 20 \
    --mask_ratio 0.75 \
    --blr 1e-2 \
    --norm_pix_loss \
    --optimizer_type 'sgd' \
    --classifier_depth 12 \
    --head_type "vit_head" \
    --single_crop \
    --output_dir "$OUTPUT_DIR_BASE/$DATASET/" \
    --dist_url "env://" \
    --finetune_mode 'encoder' \
    --resume_model ${RESUME_MODEL} \
    --resume_finetune ${RESUME_FINETUNE}

# batch 128 didn't work