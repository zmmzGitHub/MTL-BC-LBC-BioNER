from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from models.crf import *
from models.lm_lstm_crf import *
import models.utils as utils
from models.evaluator import eval_wc
from models.predictor import predict_wc  # NEW

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools
import random


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    # 1：wikipedia-pubmed-and-PMC-w2v.bin 2：PubMed-shuffle-win-30.bin
    parser.add_argument('--emb_file', default='data/PubMed-shuffle-win-30.bin',
                        help='path to pre-trained embedding')
    parser.add_argument('--train_file', nargs='+', default='./data/ner2003/eng.train.iobes',
                        help='path to training file')
    parser.add_argument('--test_file', nargs='+', default='./data/ner2003/eng.testb.iobes', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=300, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=100, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false',
                        help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    parser.add_argument('--co_train', action='store_true',
                        help='cotrain language model')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true',
                        help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--output_annotation', action='store_true', help='output annotation results or not')
    args = parser.parse_args()

    if_cuda = torch.cuda.is_available()
    if args.gpu >= 0 and if_cuda:
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)

    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    file_num = len(args.train_file)
    lines = []
    test_lines = []
    for i in range(file_num):
        with codecs.open(args.train_file[i], 'r', 'utf-8') as f:
            lines0 = f.readlines()
        lines.append(lines0)

    for i in range(file_num):
        with codecs.open(args.test_file[i], 'r', 'utf-8') as f:
            test_lines0 = f.readlines()
        test_lines.append(test_lines0)

    dataset_loader = []
    test_dataset_loader = []
    f_map = dict()
    l_map = []
    for i in range(file_num):
        l_map.append(dict())
    char_count = dict()
    train_features = []
    test_features = []
    train_labels = []
    test_labels = []
    train_features_tot = []
    test_word = []
    test_label = []
    for i in range(file_num):
        test_features0, test_labels0 = utils.read_corpus(test_lines[i])

        test_features.append(test_features0)
        test_labels.append(test_labels0)

        if args.output_annotation:  # NEW
            test_word0, test_label0 = utils.read_features_labels(test_lines[i])
            test_word.append(test_word0)
            test_label.append(test_label0)

        if args.load_check_point:
            if os.path.isfile(args.load_check_point):
                print("loading checkpoint: '{}'".format(args.load_check_point))
                checkpoint_file = torch.load(args.load_check_point)
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                c_map = checkpoint_file['c_map']
                in_doc_words = checkpoint_file['in_doc_words']
                train_features, train_labels = utils.read_corpus(lines[i])
            else:
                print("no checkpoint found at: '{}'".format(args.load_check_point))
        else:
            print('constructing coding table')
            train_features0, train_labels0, f_map, l_map[i], char_count = utils.generate_corpus_char(lines[i], f_map,
                                                                                                  l_map[i], char_count,
                                                                                                  c_thresholds=args.mini_count,
                                                                                                  if_shrink_w_feature=False)

        train_features.append(train_features0)
        train_labels.append(train_labels0)

        train_features_tot += train_features0

    shrink_char_count = [k for (k, v) in iter(char_count.items()) if v >= args.mini_count]
    char_map = {shrink_char_count[ind]: ind for ind in range(0, len(shrink_char_count))}

    char_map['<u>'] = len(char_map)  # unk for char
    char_map[' '] = len(char_map)  # concat for char
    char_map['\n'] = len(char_map)  # eof for char

    f_set = {v for v in f_map}
    dt_f_set = f_set
    f_map = utils.shrink_features(f_map, train_features_tot, args.mini_count)

    for i in range(file_num):
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features[i]), dt_f_set)
        l_set = set()
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels[i]), l_set)
        for label in l_set:
            if label not in l_map[i]:
                print("add test label...")
                l_map[i][label] = len(l_map[i])

    l_map_len = [len(m) for m in l_map]
    if not args.rand_embedding:
        print("feature size: '{}'".format(len(f_map)))
        print('loading embedding')
        if args.fine_tune:  # which means does not do fine-tune which means no <unk>
            f_map = {'<eof>': 0}
        f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, b' ', f_map, dt_f_set,
                                                                         args.caseless, args.unk, args.word_dim,
                                                                         shrink_to_corpus=args.shrink_embedding)
        print("embedding size: '{}'".format(len(f_map)))


    print('constructing dataset')

    for i in range(file_num):
        # construct dataset
        dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(
            train_features[i], train_labels[i], l_map[i], char_map, f_map, args.caseless)
        test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(
            test_features[i], test_labels[i], l_map[i], char_map, f_map, args.caseless)

        dataset_loader.append(
            [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset])
        test_dataset_loader.append(
            [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset])

    all_train_data_iter = [itertools.cycle(itertools.chain.from_iterable(data)) for data in dataset_loader]
    all_train_len = [functools.reduce(lambda x, y: x + y, list(map(lambda t: len(t), data))) for data in dataset_loader]
    print('all len: ', all_train_len)   # the total number of batches

    # build model  large_CRF is True
    print('building model')
    ner_model = LM_LSTM_CRF_sep(l_map_len, len(char_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim,
                            args.word_hidden, args.word_layers, len(f_map), args.drop_out, file_num,
                            large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words,
                            highway_layers=args.highway_layers)

    if args.load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_word_embedding(embedding_tensor)
        ner_model.rand_init(init_word_embedding=args.rand_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    crit_ner = []
    for i in range(file_num):
        crit_ner.append(CRFLoss_vb(l_map_len[i], l_map[i]['<start>'], l_map[i]['<pad>']))

    packer = []
    if args.gpu >= 0 and if_cuda is True:
        crit_lm.cuda()
        ner_model.cuda()
        for i in range(file_num):
            crit_ner[i].cuda()
            packer.append(CRFRepack_WC(l_map_len[i], True))
    else:
        if_cuda = False
        for i in range(file_num):
            packer.append(CRFRepack_WC(l_map_len[i], False))

    tot_length = sum(all_train_len)

    best_f1_dev = []
    best_f1_test = []
    for i in range(file_num):
        best_f1_dev.append(float('-inf'))
        best_f1_test.append(float('-inf'))

    best_pre_dev = []
    best_pre_test = []
    for i in range(file_num):
        best_pre_dev.append(float('-inf'))
        best_pre_test.append(float('-inf'))

    best_rec_dev = []
    best_rec_test = []
    for i in range(file_num):
        best_rec_dev.append(float('-inf'))
        best_rec_test.append(float('-inf'))

    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)

    evaluator = []
    predictor = []
    for i in range(file_num):
        evaluator.append(eval_wc(packer[i], l_map[i], args.eva_matrix))
        predictor.append(predict_wc(if_cuda, f_map, char_map, l_map[i], f_map['<eof>'], char_map['\n'],
                                    l_map[i]['<pad>'], l_map[i]['<start>'], True, args.batch_size, args.caseless))  # NEW

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        # shuffle batches
        all_indices = []
        for i in range(file_num):
            all_indices += [i] * all_train_len[i]
        random.shuffle(all_indices)

        epoch_loss = 0
        ner_model.train()

        for i in tqdm(range(len(all_indices)), mininterval=2,
                      desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):
            file_no = all_indices[i]
            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v = next(all_train_data_iter[file_no])

            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer[file_no].repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)

            ner_model.zero_grad()

            scores = ner_model(f_f, f_p, b_f, b_p, w_f, file_no)
            loss = crit_ner[file_no](scores, tg_v, mask_v)

            epoch_loss += utils.to_scalar(loss)
            if args.co_train:
                cf_p = f_p[0:-1, :].contiguous()
                cb_p = b_p[1:, :].contiguous()
                cf_y = w_f[1:, :].contiguous()
                cb_y = w_f[0:-1, :].contiguous()
                cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                loss = loss + args.lambda0 * crit_lm(cfs, cf_y.view(-1))
                cbs, _ = ner_model.word_pre_train_backward(b_f, cb_p)
                loss = loss + args.lambda0 * crit_lm(cbs, cb_y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm(ner_model.parameters(), args.clip_grad)
            optimizer.step()
            torch.cuda.empty_cache()    # empty cache


        epoch_loss /= tot_length

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # eval & save check_point
        for file_no in range(file_num):
            test_f1, test_pre, test_rec, test_acc = evaluator[file_no].calc_score(ner_model, test_dataset_loader[file_no],
                                                                         file_no)
            if test_f1 > best_f1_test[file_no]:
                patience_count = 0
                best_f1_test[file_no] = test_f1
                best_pre_test[file_no] = test_pre
                best_rec_test[file_no] = test_rec

                track_list.append(
                    {'loss': epoch_loss, 'test_f1': test_f1, 'test_acc': test_acc})

                print(
                    '(loss: %.4f, epoch: %d, dataset: %d, F1 on test = %.4f, pre on test= %.4f, rec on test= %.4f), saving...' %
                    (epoch_loss,
                     args.start_epoch,
                     file_no,
                     test_f1,
                     test_pre,
                     test_rec))

                if args.output_annotation:  # NEW
                    print('annotating')
                    with open('output_3th_sep_coh_' + str(file_num) + '_' + str(file_no) + '.txt', 'w') as fout:
                        predictor[file_no].output_batch(ner_model, test_word[file_no], test_label[file_no], fout, file_no)

                try:
                    utils.save_checkpoint({
                        'epoch': args.start_epoch,
                        'state_dict': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': f_map,
                        'l_map': l_map,
                        'c_map': char_map,
                        'in_doc_words': in_doc_words
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, args.checkpoint + 'cwlm_lstm_crf')
                except Exception as inst:
                    print(inst)

            else:
                print('(loss: %.4f, epoch: %d, dataset: %d, test F1 = %.4f, test pre = %.4f, test rec = %.4f)' %
                      (epoch_loss,
                       args.start_epoch,
                       file_no,
                       test_f1,
                       test_pre,
                       test_rec))
                track_list.append({'loss': epoch_loss, 'test_f1': test_f1, 'test_acc': test_acc})


        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')


    print('best value:')
    for i in range(file_num):
        print('dataset: %d, F1 on test: %.4f, pre on test: %.4f, rec on test: %.4f'
              % (i, best_f1_test[i], best_pre_test[i], best_rec_test[i]))