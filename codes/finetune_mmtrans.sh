# Finetuning the model

# pls refer to fairseq documentaion to know more about each of these options (https://fairseq.readthedocs.io/en/latest/command_line_tools.html)


# some notable args:
# --max-update=1000     -> for this example, to demonstrate how to finetune we are only training for 1000 steps. You should increase this when finetuning
# --arch=transformer_4x -> we use a custom transformer model and name it transformer_4x (4 times the parameter size of transformer  base)
# --user_dir            -> we define the custom transformer arch in model_configs folder and pass it as an argument to user_dir for fairseq to register this architechture
# --lr                  -> learning rate. From our limited experiments, we find that lower learning rates like 3e-5 works best for finetuning.
# --restore-file        -> reload the pretrained checkpoint and start training from here (change this path for indic-en. Currently its is set to en-indic)
# --reset-*             -> reset and not use lr scheduler, dataloader, optimizer etc of the older checkpoint
# --max_tokns           -> this is max tokens per batch


# fairseq-train ../datasets/dataset/final_bin \
# --max-source-positions=210 \
# --max-target-positions=210 \
# --max-update=10000 \
# --save-interval=1 \
# --arch=transformer_4x \
# --criterion=label_smoothed_cross_entropy \
# --source-lang=SRC \
# --lr-scheduler=inverse_sqrt \
# --target-lang=TGT \
# --label-smoothing=0.1 \
# --optimizer adam \
# --adam-betas "(0.9, 0.98)" \
# --clip-norm 1.0 \
# --warmup-init-lr 1e-07 \
# --warmup-updates 4000 \
# --dropout 0.3 \
# --tensorboard-logdir ../datasets/dataset/tensorboard-wandb \
# --save-dir ../datasets/dataset/model \
# --keep-last-epochs 5 \
# --patience 5  --seed 42 \
# --skip-invalid-size-inputs-valid-test \
# --memory-efficient-fp16 \
# --user-dir model_configs \
# --update-freq=2 \
# --distributed-world-size 1 \
# --max-tokens 256 \
# --lr 3e-5 \
# --restore-file en-indic/model/checkpoint_best.pt \
# --reset-lr-scheduler \
# --reset-meters \
# --reset-dataloader \
# --reset-optimizer
savedir=$1
image_feat_path=~/scripts/datasets/image_feats_full/vit_base_patch16_224
image_feat_dim=768
fairseq-train ../datasets/dataset/final_bin \
--max-source-positions=210 \
--max-target-positions=210 \
--max-update=20000 \
--save-interval=1 \
--arch=image_multimodal_transformer_top_4x \
--task image_mmt \
--criterion=label_smoothed_cross_entropy \
--source-lang=SRC \
--lr-scheduler=inverse_sqrt \
--target-lang=TGT \
--label-smoothing=0.1 \
--optimizer adam --train-subset train \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim --SA-image-dropout 0.3 \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 --train-subset train \
--dropout 0.3 \
--tensorboard-logdir ../datasets/dataset/tensorboard-wandb \
--save-dir $savedir \
--keep-last-epochs 3 \
--patience 5  --seed 42 \
--skip-invalid-size-inputs-valid-test \
--memory-efficient-fp16 \
--update-freq=2 \
--distributed-world-size 1 --save-interval-updates 500 \
--max-tokens 256 \
--lr 3e-5 \
--restore-file en-indic/model/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer


#Simple finetune

# fairseq-train ../datasets/dataset/final_bin \
# --max-source-positions=210 \
# --max-target-positions=210 \
# --max-update=20000 \
# --save-interval=1 \
# --arch=transformer_4x \
# --task translation \
# --criterion=label_smoothed_cross_entropy \
# --source-lang=SRC \
# --lr-scheduler=inverse_sqrt \
# --target-lang=TGT \
# --label-smoothing=0.1 \
# --optimizer adam --train-subset train \
# --adam-betas "(0.9, 0.98)" \
# --clip-norm 1.0  \
# --warmup-init-lr 1e-07 \
# --warmup-updates 4000 \
# --dropout 0.3 \
# --tensorboard-logdir ../datasets/dataset/tensorboard-wandb \
# --save-dir $savedir \
# --keep-last-epochs 3 \
# --patience 5  --seed 42 \
# --skip-invalid-size-inputs-valid-test \
# --memory-efficient-fp16 \
# --update-freq=2 \
# --distributed-world-size 1 --save-interval-updates 500 \
# --max-tokens 256 \
# --lr 3e-5 \
# --restore-file en-indic/model/checkpoint_best.pt \
# --reset-lr-scheduler \
# --reset-meters \
# --reset-dataloader \
# --reset-optimizer