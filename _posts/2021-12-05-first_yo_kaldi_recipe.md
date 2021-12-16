
# Kaldi recipe for Yorùbá

Today we created a [Kaldi](https://github.com/Niger-Volta-LTI/kaldi/tree/egs-yoruba-recipe) recipe for Yorùbá based on the following speech datasets: 
 * [SLR86](https://www.openslr.org/86/) for training 
 * [Lagos-NWU](https://repo.sadilar.org/handle/20.500.12185/431) for evaluation. 

The recipe is a fork of [Mini LibriSpeech](https://www.openslr.org/31/) that follows the traditional HMM-GMM low-data recipe, bootstrapping alignments from a monophone model and then proceeding to train a delta-delta triphone system, followed by LDA+MLLT, LDA+MLLT+SAT finally doing language-model (LM) rescoring with a "big" 4gram model ... to get a grand WER evaluation of 80%.

I don't think I've ever been as excited for lift-off with such a high-error rate. What it represents rather is a working platform in Kaldi for experimentation, making baselines. Next, we shall do lots of decoding of untranscribed Yorùbá audio for error analysis. There are many vectors for improvement including data augmentation, training beyond a simple triphone model, say attempt a LF-MMI training on a GPU ... but the basics are in place.

## Results

For reference, `tg-` stands for trigram, as in [N-gram language models](https://web.stanford.edu/~jurafsky/slp3/3.pdf). The different between large, medium and small are as follows. **tgsmall** is pruned, **tgmed** is the corpora text without any extra efforts, **tglarge** is not a trigram, but a 4-gram. These choices were made to map into the Mini LibriSpeech recipe as directly as possible and we may make different LM modeling choices in the next steps.

**tgsmall**
```
iroro:~/github/yoruba-asr/exp_baseline_01/tri3b/decode_tgsmall_dev_clean_2$ cat wer_10_0.0
compute-wer --text --mode=present ark:exp/tri3b/decode_tgsmall_dev_clean_2/scoring/test_filt.txt ark,p:-
%WER 83.38 [ 18002 / 21590, 242 ins, 10016 del, 7744 sub ]
%SER 99.44 [ 4292 / 4316 ]
Scored 4316 sentences, 0 not present in hyp.
```

**tgmed** 
```
iroro:~/github/yoruba-asr/exp_baseline_01/tri3b/decode_tgmed_dev_clean_2$ cat wer_10_0.0
compute-wer --text --mode=present ark:exp/tri3b/decode_tgmed_dev_clean_2/scoring/test_filt.txt ark,p:-
%WER 79.95 [ 17258 / 21585, 212 ins, 9862 del, 7184 sub ] [PARTIAL]
%SER 98.70 [ 4259 / 4315 ]
Scored 4315 sentences, 1 not present in hyp.
```

**tglarge** 
```
compute-wer --text --mode=present ark:exp/tri3b/decode_tglarge_dev_clean_2/scoring/test_filt.txt ark,p:-
%WER 80.00 [ 17272 / 21590, 160 ins, 10589 del, 6523 sub ]
%SER 98.31 [ 4243 / 4316 ]
Scored 4316 sentences, 0 not present in hyp.
```
