export MODEL_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="../output/CelebA-HQ/47"
export DREAMBOOTH_OUTPUT_DIR="../dreambooth_model/CelebA-HQ/47"
export CLASS_DIR="../data/class-person"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a dslr portrait of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8 \
  --enable_xformers_memory_efficient_attention \

