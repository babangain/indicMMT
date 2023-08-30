#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
ref_fname=$6

SRC_PREFIX='SRC'
TGT_PREFIX='TGT'
tgt_output_fname=$outfname
#`dirname $0`/env.sh
SUBWORD_NMT_DIR='subword-nmt'
model_dir=$exp_dir/img_finetune_crop_dropout_0.3all
data_bin_dir=$exp_dir/final_bin
input_size=4785
### normalization and script conversion

# echo "Applying normalization and script conversion"
# input_size=`python scripts/preprocess_translate.py $infname $outfname.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# ### apply BPE to input file

# echo "Applying BPE"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${SRC_PREFIX} \
#     --vocabulary $exp_dir/vocab/vocab.$SRC_PREFIX \
#     --vocabulary-threshold 5 \
#     < $outfname.norm \
#     > $outfname._bpe

# not needed for joint training
# echo "Adding language tags"
# python scripts/add_tags_translate.py $outfname._bpe $outfname.bpe $src_lang $tgt_lang

### run decoder

echo "Decoding"
echo "Model dir"
echo $model_dir
# src_input_bpe_fname=$outfname.bpe
# tgt_output_fname=$outfname
# fairseq-interactive  $data_bin_dir \
#     -s $SRC_PREFIX -t $TGT_PREFIX \
#     --distributed-world-size 1  \
#     --path $model_dir/checkpoint_best.pt \
#     --batch-size 64  --buffer-size 2500 --beam 5  --remove-bpe \
#     --skip-invalid-size-inputs-valid-test \
#     --user-dir model_configs \
#     --input $src_input_bpe_fname  >  $tgt_output_fname.log 2>&1
image_feat_path=~/scripts/datasets/image_feats/vit_base_patch16_224
image_feat_dim=768
fairseq-generate  $data_bin_dir \
    -s $SRC_PREFIX -t $TGT_PREFIX --task image_mmt \
    --distributed-world-size 1  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim \
    --path $model_dir/checkpoint_best.pt --gen-subset test \
    --batch-size 64 --beam 5  --remove-bpe \
    --skip-invalid-size-inputs-valid-test --output $tgt_output_fname.output  >  $tgt_output_fname.log 2>&1
# fairseq-generate  $data_bin_dir \
#     -s $SRC_PREFIX -t $TGT_PREFIX --task translation \
#     --distributed-world-size 1  \
#     --path $model_dir/checkpoint_best.pt --gen-subset test \
#     --batch-size 64 --beam 5  --remove-bpe \
#     --skip-invalid-size-inputs-valid-test --output $tgt_output_fname.output  >  $tgt_output_fname.log 2>&1
echo "Extracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.

langs=(hi bn ml)
INDIC_SCRIPT=../indic_nlp_library/indicnlp
MOSES_SCRIPT=mosesdecoder-RELEASE-2.1.1/scripts
RIBES=RIBES
for lang in ${langs[@]};do

    python scripts/postprocess_translate.py $tgt_output_fname.log $tgt_output_fname.$lang $input_size $lang true
    if [ "$lang" = "bn" ]; then
        head -n 1595 $tgt_output_fname.$lang > $tgt_output_fname.final.$lang
    fi
    if [ "$lang" = "hi" ]; then
        sed -n '1596,3190p' $tgt_output_fname.$lang > $tgt_output_fname.final.$lang
    fi
    if [ "$lang" = "ml" ]; then
        sed -n '3191,4785p' $tgt_output_fname.$lang > $tgt_output_fname.final.$lang
    fi

    echo "Results for $lang"
    cat $tgt_output_fname.final.$lang | sacrebleu ../datasets/dataset/test/test.$lang -m bleu ter 
    TASK=en-$lang
    cp $tgt_output_fname.final.$lang results.org/en-$lang.txt
    cp ../datasets/dataset/test/test.$lang tests.org/en-$lang.txt

    for file in tests results; do
       python ${INDIC_SCRIPT}/normalize/indic_normalize.py ${file}.org/${TASK}.txt ${file}.org/${TASK}.normalized.txt $lang
       python ${INDIC_SCRIPT}/tokenize/indic_tokenize.py ${file}.org/${TASK}.normalized.txt ${file}.tok/${TASK}.indic.txt $lang
    done
    perl ${MOSES_SCRIPT}/generic/multi-bleu.perl tests.tok/${TASK}.indic.txt < results.tok/${TASK}.indic.txt
    python3 ${RIBES}/RIBES.py -c -r tests.tok/${TASK}.indic.txt results.tok/${TASK}.indic.txt
done

# This block is now moved to compute_bleu.sh for release with more documentation.
# if [ $src_lang == 'en' ]; then
#     # indicnlp tokenize the output files before evaluation
#     input_size=`python scripts/preprocess_translate.py $ref_fname $ref_fname.tok $tgt_lang`
#     input_size=`python scripts/preprocess_translate.py $tgt_output_fname $tgt_output_fname.tok $tgt_lang`
#     sacrebleu --tokenize none $ref_fname.tok < $tgt_output_fname.tok
# else
#     # indic to en models
#     sacrebleu $ref_fname < $tgt_output_fname
# fi
# echo `date`



# input_size=4200
# fairseq-generate  $data_bin_dir \
#     -s $SRC_PREFIX -t $TGT_PREFIX --task image_mmt \
#     --distributed-world-size 1  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim \
#     --path $model_dir/checkpoint_best.pt --gen-subset challenge \
#     --batch-size 64 --beam 5  --remove-bpe \
#     --skip-invalid-size-inputs-valid-test --output $tgt_output_fname.output.chal  >  $tgt_output_fname.chal.log 2>&1

# input_size=4200
# # fairseq-generate  $data_bin_dir \
# #     -s $SRC_PREFIX -t $TGT_PREFIX --task translation \
# #     --distributed-world-size 1  \
# #     --path $model_dir/checkpoint_best.pt --gen-subset challenge \
# #     --batch-size 64 --beam 5  --remove-bpe \
# #     --skip-invalid-size-inputs-valid-test --output $tgt_output_fname.output.chal  >  $tgt_output_fname.chal.log 2>&1
# echo "Extracting translations, script conversion and detokenization"
# # this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.

# langs=(hi bn ml)
# for lang in ${langs[@]};do

#     python scripts/postprocess_translate.py $tgt_output_fname.chal.log $tgt_output_fname.chal.$lang $input_size $lang true
#     if [ "$lang" = "bn" ]; then
#         head -n 1400 $tgt_output_fname.chal.$lang > $tgt_output_fname.final.chal.$lang
#     fi
#     if [ "$lang" = "hi" ]; then
#         sed -n '1401,2800p' $tgt_output_fname.chal.$lang > $tgt_output_fname.final.chal.$lang
#     fi
#     if [ "$lang" = "ml" ]; then
#         sed -n '2801,4200p' $tgt_output_fname.chal.$lang > $tgt_output_fname.final.chal.$lang
#     fi

#     echo "Results for $lang"
#     cat $tgt_output_fname.final.chal.$lang | sacrebleu ../datasets/dataset/challenge/challenge.$lang -m bleu ter 
#     TASK=en-$lang
#     cp $tgt_output_fname.final.chal.$lang results.org/en-$lang.txt
#     cp ../datasets/dataset/challenge/challenge.$lang tests.org/en-$lang.txt

#     for file in tests results; do
#        python ${INDIC_SCRIPT}/normalize/indic_normalize.py ${file}.org/${TASK}.txt ${file}.org/${TASK}.normalized.txt $lang
#        python ${INDIC_SCRIPT}/tokenize/indic_tokenize.py ${file}.org/${TASK}.normalized.txt ${file}.tok/${TASK}.indic.txt $lang
#     done
#     perl ${MOSES_SCRIPT}/generic/multi-bleu.perl tests.tok/${TASK}.indic.txt < results.tok/${TASK}.indic.txt
#     python3 ${RIBES}/RIBES.py -c -r tests.tok/${TASK}.indic.txt results.tok/${TASK}.indic.txt
# done


# echo "Translation completed"
