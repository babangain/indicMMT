src='en'
tgt='hi'

TEXT=data/visgenome

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train.bpe \
  --validpref $TEXT/valid.bpe \
  --testpref $TEXT/test.bpe \
  --destdir data-bin/visgenome \
  --workers 8 --joined-dictionary --srcdict $TEXT/vocab.en
