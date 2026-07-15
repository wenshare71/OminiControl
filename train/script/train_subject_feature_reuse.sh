# 训练"独立条件版" subject LoRA(feature reuse / kv_cache 专用)
# 与 train_subject.sh 唯一差异:config 换成 subject_feature_reuse.yaml
# *[Specify the GPU devices to use]
# export CUDA_VISIBLE_DEVICES=0,1

export OMINI_CONFIG=./train/config/subject_feature_reuse.yaml

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41354 -m omini.train_flux.train_subject
