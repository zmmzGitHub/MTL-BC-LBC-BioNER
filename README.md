# MTL-BC-LBC-BioNER
Two multi-task learning approches for Bio-NER
# Dependencies
The code is written in Python 3.5. Its dependencies is summarized in ```requirements.txt``` file
# usage
```train_shuffle_sep.py``` is to train multi-task models on datasets for biomedical named entity recognition (Bio-NER).
```train_shuffle_pos.py``` is to train multi-task models on Bio-NER and POS (part-of-speech) tagging datasets.
The usages of two scripts can be accessed by
```
python train_shuffle_sep.py -h
python train_shuffle_pos.py -h
```

The default running commands are:
```
python3 train_wc.py --train_file [training file 1] [training file 2] ... [training file N] \
                    --dev_file [developing file 1] [developing file 2] ... [developing file N] \
                    --test_file [testing file 1] [testing file 2] ... [testing file N] \
                    --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200
```

Users may incorporate an arbitrary number of corpora into the training process. In each epoch, our model randomly selects one dataset _i_. We use training set _i_ to learn the parameters and developing set _i_ to evaluate the performance. If the current model achieves the best performance for dataset _i_ on the developing set, we will then calculate the precision, recall and F1 on testing set _i_.

