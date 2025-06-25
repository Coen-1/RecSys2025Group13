# Use distributed data parallel
python lightning_pretrain_MIND.py \
    --model_name_or_path allenai/longformer-base-4096 \
    --train_file ./MIND_Recformer/mind_data_recformer_small/mind_data_recformer_small_train.json \
    --dev_file ./MIND_Recformer/mind_data_recformer_small/mind_data_recformer_small_dev.json \
    --item_attr_file ./MIND_Recformer/mind_data_recformer_small/mind_data_recformer_small_meta_data.json \
    --output_dir result/recformer_pretraining \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 16  \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 1 \
    --fp16 \
    --fix_word_embedding \
    --max_steps 2000 \
    --valid_step 100