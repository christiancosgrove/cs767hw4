# CS767 HW4 : Diverse Responses and Evaluation
## Christian Cosgrove, Darius Irani, JungMin Lee

## OpenSubtitles Loading

To implement the data filtering as in the DialoGPT paper, we created a new teacher agent (`TaskDialoGPTFilteringTeacher`) that implements the filtering. Before yielding data examples, we run through the dataset and collect trigram frequencies. If an example contains a trigram that occurs more than 500 times, we remove it. Moreover, we filter all examples were the first utterance does not end in a question mark.

## Training

We trained a transformer model (`transformer/generator`) on OpenSubtitles for roughly 12 hours. We used a batch size of 10, learning rate of 1, and truncation length of 64. 

We trained the backwards model the same way. To train the backwards model, we created a new teacher deriving from our first teacher--`TaskDialoGPTFilteringReverseTeacher`--that simply reverses the question and the answer. Our backwards model is the same as the forwards model.

To finetune on DailyDialog, we simply use the `--init_model` argument. We only finetuned for a short time (~1 hour).

## MMI

MMI was straightforward to implement. We simply extended the `BeamSearch` class (see `MMISearch`). We modified the `get_rescored_finished` function, which is called to rerank the responses after they are generated using beam search. We hardcoded model loading for this (although it would be straightforward to change this).

Because the transformer models output log-probabilities rather than probabilites, computing the adjustment to the MMI objective is trivial--simply sumtract the sum of the logits from the reverse model (weighted by a parameter lambda).

## Evaluations

Our MTurk implementation is found in `parlai/mturk/tasks/dialo_eval/run.py`. We use the `dailydialog:NoStart` task, which eliminates `__SILENCE__` tokens.

In our evaluations, we found that the finetuned model performed worse than the pretrained model (paradoxically). We aren't sure why this is the case; perhaps our hyperparameter settings are causing overfitting on the finetuning dataset.

MMI doesn't seem to improve the quality of responses. We think this is because something is wrong with our reverse model. The reverse model outputs very large logits for some tokens (like `-1e20`); this might be throwing things off. If we had more time, we would investigate the reverse model training and make sure it outputted reasonable log-probabilities.

