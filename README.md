# MTL-BC-LBC-BioNER
Two multi-task learning approches for Bio-NER
# Dependencies and References
The code is written in Python 3.5. Its dependencies is summarized in ```requirements.txt``` file
We adapt some code from: <br>
https://github.com/LiyuanLucasLiu/LM-LSTM-CRF <br>
https://github.com/yuzhimanhua/lm-lstm-crf  <br>
# Usage
```train_shuffle_sep.py``` is to train multi-task models on datasets for biomedical named entity recognition (Bio-NER).
```train_shuffle_pos.py``` is to train multi-task models on Bio-NER and POS (part-of-speech) tagging datasets.

The usages of two scripts can be accessed by

```
python train_shuffle_sep.py -h
python train_shuffle_pos.py -h
```

The commands for MTL-BC and MTL-LBC on Bio-NER datasets are:
```
python3 train_shuffle_sep.py --train_file [training file 1] [training file 2] ... [training file N] \
                             --test_file [testing file 1] [testing file 2] ... [testing file N] \
                             --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200
```
```
python3 train_shuffle_sep.py --train_file [training file 1] [training file 2] ... [training file N] \
                             --test_file [testing file 1] [testing file 2] ... [testing file N] \
                             --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200 --co_train --high_way
```
The commands for MTL-BC and MTL-LBC both on Bio-NER and POS datasets are:
```
python3 train_shuffle_pos.py --train_file [training bio-ner file 1] ... [training bio-ner file N] [training pos file] \
                             --test_file [testing bio-ner file 1] ... [testing bio-ner file N] [testing pos file] \
                             --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200
```
```
python3 train_shuffle_pos.py --train_file [training bio-ner file 1] ... [training bio-ner file N] [training pos file] \
                             --test_file [testing bio-ner file 1] ... [testing bio-ner file N] [testing pos file]\
                             --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200 --co_train --high_way
```

Users may incorporate an arbitrary number of corpora into the training process.
## Note
We merge training and development sets before running these two scripts. For ```train_shuffle_pos.py```, the path of POS dataset is set behind all Bio-NER datasets. 

