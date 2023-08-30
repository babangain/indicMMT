#!/usr/bin/bash
set -e

_mask=no_mask
_image_feat=vit_base_patch16_224

# set device
gpu=3

model_root_dir=checkpoints

# set task
task=multi30k-en2de
mask_data=$_mask
image_feat=$_image_feat

who=test	#test1, test2
random_image_translation=0 #1
length_penalty=0.8

# set tag
model_dir_tag=$image_feat/$mask_data

# if [ $task == "multi30k-en2de" ]; then
# 	tgt_lang=de
# 	if [ $mask_data == "mask0" ]; then
# 	        data_dir=multi30k.en-de
# 	elif [ $mask_data == "mask1" ]; then
# 	        data_dir=multi30k.en-de.mask1
# 	elif [ $mask_data == "mask2" ]; then
# 	        data_dir=multi30k.en-de.mask2
# 	elif [ $mask_data == "mask3" ]; then
# 	        data_dir=multi30k.en-de.mask3
# 	elif [ $mask_data == "mask4" ]; then
# 	        data_dir=multi30k.en-de.mask4
# 	elif [ $mask_data == "maskc" ]; then
# 	        data_dir=multi30k.en-de.maskc
# 	elif [ $mask_data == "maskp" ]; then
# 	        data_dir=multi30k.en-de.maskp
# 	fi
# elif [ $task == 'multi30k-en2fr' ]; then
# 	tgt_lang=fr
# 	if [ $mask_data == "mask0" ]; then
#         	data_dir=multi30k.en-fr
# 	elif [ $mask_data == "mask1" ]; then
# 	        data_dir=multi30k.en-fr.mask1
# 	elif [ $mask_data == "mask2" ]; then
#       		data_dir=multi30k.en-fr.mask2
# 	elif [ $mask_data == "mask3" ]; then
# 	        data_dir=multi30k.en-fr.mask3
# 	elif [ $mask_data == "mask4" ]; then
# 	        data_dir=multi30k.en-fr.mask4
# 	elif [ $mask_data == "maskc" ]; then
# 	        data_dir=multi30k.en-fr.maskc
# 	elif [ $mask_data == "maskp" ]; then
# 	        data_dir=multi30k.en-fr.maskp
# 	fi
# fi

# if [ $image_feat == "vit_tiny_patch16_384" ]; then
# 	image_feat_path=data/$image_feat
# 	image_feat_dim=192
# elif [ $image_feat == "vit_small_patch16_384" ]; then
# 	image_feat_path=data/$image_feat
# 	image_feat_dim=384
# elif [ $image_feat == "vit_base_patch16_384" ]; then
# 	image_feat_path=data/$image_feat
# 	image_feat_dim=768
# elif [ $image_feat == "vit_large_patch16_384" ]; then
# 	image_feat_path=data/$image_feat
# 	image_feat_dim=1024
# fi

# data set
# ensemble=5
batch_size=128
beam=5
src_lang='en'
tgt_lang='hi'
image_feat=vit_base_patch16_224
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/visgenome/$tag.freeze
src_lang=en
image_feat_dim=768
image_feat_path=data/$image_feat
data_dir=visgenome
model_dir=$model_root_dir/$task/$model_dir_tag
echo $model_dir
checkpoint=checkpoint_best.pt

# if [ -n "$ensemble" ]; then
#         if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
#                 PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
#         fi
#         checkpoint=last$ensemble.ensemble.pt
# fi

# output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate ~/scripts/chat_en_hi/data/wat2022_data/data_bin/visgenome 
  -s en -t hi 
  --path ~/scripts/multimodal/fairseq_mmt/checkpoints/visgenome/vit_base_patch16_224/vit_base_patch16_224-no_mask.freeze/checkpoint_best.pt  
  --gen-subset $who 
  --batch-size $batch_size --beam $beam  
  --remove-bpe
  --task image_mmt
  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --output ~/scripts/multimodal/fairseq_mmt/checkpoints/visgenome/vit_base_patch16_224/vit_base_patch16_224-no_mask.freeze/hypo.txt" 

if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi

cmd=${cmd}" | tee "${output}
eval $cmd
MOSES_DIR=/Data/baban/scripts/mosesdecoder
python3 rerank.py ~/scripts/multimodal/fairseq_mmt/checkpoints/visgenome/vit_base_patch16_224/vit_base_patch16_224-no_mask.freeze/hypo.txt ~/scripts/multimodal/fairseq_mmt/checkpoints/visgenome/vit_base_patch16_224/vit_base_patch16_224-no_mask.freeze/hypo.sorted
OUTFILENAME=/Data/baban/scripts/multimodal/fairseq_mmt/checkpoints/visgenome/vit_base_patch16_224/vit_base_patch16_224-no_mask.freeze/hypo.sorted
cat $OUTFILENAME | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi
wc -l $OUTFILENAME.hi
cat $OUTFILENAME.hi | sacrebleu ~/scripts/chat_en_hi/data/wat2022_data/data/visgenome/test.hi  -m bleu ter

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k/test.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k/test.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k/test.coco.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.fr
fi	

hypo=$model_dir/hypo.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log

# cal gate, follow Revisit-MMT
#python3 scripts/visual_awareness.py --input $model_dir_tag/gated.txt 

# cal accurary
python3 cal_acc.py $hypo $who $task
