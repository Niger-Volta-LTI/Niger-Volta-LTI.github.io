
# Kaldi recipe for Yorùbá

Today we created a [Kaldi](https://github.com/Niger-Volta-LTI/kaldi/tree/egs-yoruba-recipe) recipe for Yorùbá based on the following speech datasets: SLR86, for training & Lagos-NWU for evaluation. 

The recipe is a fork of mini-librispeech that follows the traditional HMM-GMM low data recipe, bootstrapping alignments from a monophone model and then proceeding to train a delta-delta triphone system, followed by LDA+MLLT, LDA+MLLT+SAT finally doing language-model (LM) rescoring with a "big" 4gram model ... to get a grand WER evaluation of 80%.

I don't think I've ever been as excited for lift-off with such a high-error rate. What it represents rather is lift-off and a working platform in Kaldi for experimentation, making baselines. Next we shall do lots of decoding of untranscribed Yorùbá for error analysis. There are many vectors for improvement including data augmentation, training beyond a simple triphone model, say attempt a LF-MMI training on a GPU augmentation ... but the basics are in place.
