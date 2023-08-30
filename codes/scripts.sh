pip install torch==1.7.1+cu113 torchvision==0.8.2+cu113 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html



python scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path /Data/shared/visual/combined_images/
python scripts/get_img_feat.py --dataset valid --model vit_base_patch16_384 --path /Data/shared/visual/combined_images/
python scripts/get_img_feat.py --dataset challenge --model vit_base_patch16_384 --path /Data/shared/visual/combined_images/
python scripts/get_img_feat.py --dataset test --model vit_base_patch16_384 --path /Data/shared/visual/combined_images/

mask_data=no_mask
src_lang='en_XX'
tgt_lang='hi_IN'
image_feat=vit_base_patch16_384
tag=$image_feat/$mask_data
save_dir=checkpoints/multi30k-en2de/$tag
image_feat_path=data/$image_feat
image_feat_dim=768

criterion=label_smoothed_cross_entropy
fp16=1
lr=0.005
warmup=2000
max_tokens=4096
update_freq=1
keep_last_epochs=5
patience=10
max_update=8000
dropout=0.3

arch=image_multimodal_transformer_SA
SA_attention_dropout=0.1
SA_image_dropout=0.1
SA_text_dropout=0

export CUDA_VISIBLE_DEVICES=2
nohup fairseq-train ~/scripts/multimodal/vita/fairseq/data/postprocessed/final/en-hi --save-dir $save_dir.full_SA \
  --distributed-world-size 1 -s $src_lang -t $tgt_lang \
  --arch $arch \
  --dropout $dropout \
  --criterion $criterion --label-smoothing 0.1 \
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update \
  --find-unused-parameters  --share-all-embeddings --patience $patience \
  --keep-last-epochs $keep_last_epochs \
  --SA-image-dropout $SA_image_dropout \
  --SA-attention-dropout $SA_attention_dropout \
  --SA-text-dropout $SA_text_dropout > nohup_full_img_mmt_SA.out &

bssh translate_mmt.sh no_mask vit_base_patch16_384 