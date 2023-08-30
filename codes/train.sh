# python scripts/get_img_feat.py --dataset valid --model vit_base_patch16_224 --path /Data/shared/visual/combined_images



mask_data=no_mask
data_dir=visgenome
src_lang='en'
tgt_lang='hi'
image_feat=vit_base_patch16_224
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/visgenome/$tag
image_feat_path=data/$image_feat
image_feat_dim=768

criterion=label_smoothed_cross_entropy
fp16=1
lr=0.00005
max_tokens=2000
update_freq=1
keep_last_epochs=1
patience=5
max_update=5000
dropout=0.0

arch=image_multimodal_transformer_SA_top_samanantar
SA_attention_dropout=0.0
SA_image_dropout=0.0
SA_text_dropout=0

CUDA_VISIBLE_DEVICES=6 fairseq-train data-bin/$data_dir \
  --save-dir $save_dir \
  --distributed-world-size 1 -s $src_lang -t $tgt_lang \
  --arch $arch  --memory-efficient-fp16 \
  --dropout $dropout \
  --criterion $criterion \
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim --seed 42 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr  --lr-scheduler inverse_sqrt \
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update \
  --find-unused-parameters \
  --share-all-embeddings \
  --ddp-backend=no_c10d \
  --patience $patience \
  --keep-last-epochs $keep_last_epochs \
  --SA-image-dropout $SA_image_dropout \
  --SA-attention-dropout $SA_attention_dropout \
  --SA-text-dropout $SA_text_dropout \
  --finetune-from-model /Data/baban/scripts/chat_en_hi/data/wat2022_data/models/samanantar.new/checkpoint_last.pt