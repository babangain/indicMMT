exp_dir=../datasets/noise
src_lang=en
tgt_lang=indic

# change this to indic-en, if you have downloaded the indic-en dir or m2m if you have downloaded the indic2indic model
download_dir=en-indic

train_data_dir=$exp_dir/train
valid_data_dir=$exp_dir/valid
test_data_dir=$exp_dir/test
challenge_data_dir=$exp_dir/challenge

echo $exp_dir



src_lang=en
tgt_lang=indic



echo "Running experiment ${exp_dir} on ${src_lang} to ${tgt_lang}"


train_processed_dir=$exp_dir/data
validtest_processed_dir=$exp_dir/data

out_data_dir=$exp_dir/final_bin

mkdir -p $train_processed_dir
mkdir -p $validtest_processed_dir
mkdir -p $out_data_dir

# indic languages.
# cvit-pib corpus does not have as (assamese) and kn (kannada), hence its not part of this list
langs=(hi bn ml)

for lang in ${langs[@]};do
	if [ $src_lang == en ]; then
		tgt_lang=$lang
	else
		src_lang=$lang
	fi

	train_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	validtest_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	mkdir -p $train_norm_dir
	mkdir -p $validtest_norm_dir


    # preprocessing pretokenizes the input (we use moses tokenizer for en and indicnlp lib for indic languages)
    # after pretokenization, we use indicnlp to transliterate all the indic data to validnagiri script

	# train preprocessing
	train_infname_src=$train_data_dir/en-${lang}/train.$src_lang
	train_infname_tgt=$train_data_dir/en-${lang}/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang
	echo "Applying normalization and script conversion for train $lang"
	input_size=`python scripts/preprocess_translate.py $train_infname_src $train_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $train_infname_tgt $train_outfname_tgt $tgt_lang true`
	echo "Number of sentences in train $lang: $input_size"

	# valid preprocessing
	valid_infname_src=$valid_data_dir/valid.$src_lang
	valid_infname_tgt=$valid_data_dir/valid.$tgt_lang
	valid_outfname_src=$validtest_norm_dir/valid.$src_lang
	valid_outfname_tgt=$validtest_norm_dir/valid.$tgt_lang
	echo "Applying normalization and script conversion for valid $lang"
	input_size=`python scripts/preprocess_translate.py $valid_infname_src $valid_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $valid_infname_tgt $valid_outfname_tgt $tgt_lang true`
	echo "Number of sentences in valid $lang: $input_size"

	# test preprocessing
	test_infname_src=$test_data_dir/test.$src_lang
	test_infname_tgt=$test_data_dir/test.$tgt_lang
	test_outfname_src=$validtest_norm_dir/test.$src_lang
	test_outfname_tgt=$validtest_norm_dir/test.$tgt_lang
	echo "Applying normalization and script conversion for test $lang"
	input_size=`python scripts/preprocess_translate.py $test_infname_src $test_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $test_infname_tgt $test_outfname_tgt $tgt_lang true`
	echo "Number of sentences in test $lang: $input_size"

	# challenge preprocessing
	challenge_infname_src=$challenge_data_dir/challenge.$src_lang
	challenge_infname_tgt=$challenge_data_dir/challenge.$tgt_lang
	challenge_outfname_src=$validtest_norm_dir/challenge.$src_lang
	challenge_outfname_tgt=$validtest_norm_dir/challenge.$tgt_lang
	echo "Applying normalization and script conversion for challenge $lang"
	input_size=`python scripts/preprocess_translate.py $challenge_infname_src $challenge_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $challenge_infname_tgt $challenge_outfname_tgt $tgt_lang true`
	echo "Number of sentences in challenge $lang: $input_size"
done




# Now that we have preprocessed all the data, we can now merge these different text files into one
# ie. for en-as, we have train.en and corresponding train.as, similarly for en-bn, we have train.en and corresponding train.bn
# now we will concatenate all this into en-X where train.SRC will have all the en (src) training data and train.TGT will have all the concatenated indic lang data

python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'train'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'valid'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'test'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'challenge'

# use the vocab from downloaded dir
cp -r $download_dir/vocab $exp_dir


echo "Applying bpe to the new finetuning data"
bash apply_single_bpe_traindevtest_notag.sh $exp_dir

mkdir -p $exp_dir/final

# We also add special tags to indicate the source and target language in the inputs
#  Eg: to translate a sentence from english to hindi , the input would be   __src__en__   __tgt__hi__ <en bpe tokens>

echo "Adding language tags"
python scripts/add_joint_tags_translate.py $exp_dir 'train'
python scripts/add_joint_tags_translate.py $exp_dir 'valid'
python scripts/add_joint_tags_translate.py $exp_dir 'test'
python scripts/add_joint_tags_translate.py $exp_dir 'challenge'



data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

rm -rf $out_data_dir

# binarizing the new data (train, valid and test) using dictionary from the download dir

 num_workers=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`

data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

# rm -rf $out_data_dir

echo "Binarizing data. This will take some time depending on the size of finetuning data"
fairseq-preprocess --source-lang SRC --target-lang TGT \
 --trainpref $data_dir/train --validpref $data_dir/valid --testpref $data_dir/test --challengepref $data_dir/challenge \
 --destdir $out_data_dir --workers $num_workers \
 --srcdict $download_dir/final_bin/dict.SRC.txt --tgtdict $download_dir/final_bin/dict.TGT.txt --thresholdtgt 5 --thresholdsrc 5