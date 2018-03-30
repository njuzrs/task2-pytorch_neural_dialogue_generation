# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 15:36:19 2018

@author: zrssch
"""
from __future__ import division
from __future__ import print_function

import os
import sys
import utils.data_utils as data_utils
from seq2seq import Encoder, Decoder, Seq2Seq
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
import random
import numpy as np
import math
import pickle

sys.path.append('../utils')

def read_data(config, source_path, target_path, max_size=None):
    data_set = []
    source_file = open(source_path, 'r')
    target_file = open(target_path, 'r')
    source, target = source_file.readline(), target_file.readline()
    counter = 0
    while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
            print("  reading disc_data line %d" % counter)
            sys.stdout.flush()
        source_ids = [int(x) for x in source.strip().split()]
        target_ids = [int(x) for x in target.strip().split()]
        target_ids.append(data_utils.EOS_ID)
        if max(len(source_ids), len(target_ids)) < config.maxlen:
            data_set.append([source_ids, target_ids])
        source, target = source_file.readline(), target_file.readline()
    source_file.close()
    target_file.close()
    return data_set
    
def prepare_data(gen_config):
    if os.path.exists('vocab') and os.path.exists('rev_vocab') and os.path.exists('dev_set') and os.path.exists('train_set'):
        fr_vocab = open('vocab','rb') 
        fr_rev_vocab = open('rev_vocab','rb') 
        fr_dev_set = open('dev_set','rb') 
        fr_train_set = open('train_set','rb')
        vocab = pickle.load(fr_vocab)
        rev_vocab = pickle.load(fr_rev_vocab)
        dev_set = pickle.load(fr_dev_set)
        train_set = pickle.load(fr_train_set)
        fr_vocab.close()
        fr_rev_vocab.close()
        fr_dev_set.close()
        fr_train_set.close()
    else:
        train_path = os.path.join(gen_config.train_dir, "chitchat.train")
        voc_file_path = [train_path+".answer", train_path+".query"]
        vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
        data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    
        print("Preparing Chitchat gen_data in %s" % gen_config.train_dir)
        train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
            gen_config.train_dir, vocab, gen_config.vocab_size)
    
        # Read disc_data into buckets and compute their sizes.
        print ("Reading development and training gen_data (limit: %d)."
                   % gen_config.max_train_data_size)
        dev_set = read_data(gen_config, dev_query, dev_answer)
        train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)
        
        fw_vocab = open('vocab','wb') 
        fw_rev_vocab = open('rev_vocab','wb') 
        fw_dev_set = open('dev_set','wb') 
        fw_train_set = open('train_set','wb') 
        pickle.dump(vocab, fw_vocab)
        pickle.dump(rev_vocab, fw_rev_vocab)
        pickle.dump(dev_set, fw_dev_set)
        pickle.dump(train_set, fw_train_set)
        fw_vocab.close()
        fw_rev_vocab.close()
        fw_dev_set.close()
        fw_train_set.close()
    return vocab, rev_vocab, dev_set, train_set

def prepare_data_new(gen_config):
    if os.path.exists('vocab') and os.path.exists('rev_vocab') and os.path.exists('dev_set') and os.path.exists('train_set'):
        fr_vocab = open('vocab','rb') 
        fr_rev_vocab = open('rev_vocab','rb') 
        fr_dev_set = open('dev_set','rb') 
        fr_train_set = open('train_set','rb')
        vocab = pickle.load(fr_vocab)
        rev_vocab = pickle.load(fr_rev_vocab)
        dev_set = pickle.load(fr_dev_set)
        train_set = pickle.load(fr_train_set)
        fr_vocab.close()
        fr_rev_vocab.close()
        fr_dev_set.close()
        fr_train_set.close()
    else:
        '''
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).
        '''
        train_set = []
        dev_set = []
        train_path = os.path.join(gen_config.train_dir, "t_given_s_dialogue_length2_6.txt")
        vocab_path = os.path.join(gen_config.train_dir, "movie_25000")
        fr1 = open(train_path, 'r')
        fr2 = open(vocab_path, 'r')
        line = fr1.readline()
        counter = 0
        while line:
            counter += 1
            if counter % 100000 == 0:
                print("  reading disc_data line %d" % counter)
                sys.stdout.flush()
            source_target = line.strip().split('|')
            source = source_target[0]
            target = source_target[1]        
            source_ids = [int(x) for x in source.strip().split()]
            target_ids = [int(x) for x in target.strip().split()]
            target_ids.append(data_utils.EOS_ID)
            if max(len(source_ids), len(target_ids)) < gen_config.maxlen:
                if counter%100==0:
                    dev_set.append([source_ids, target_ids])
                else:
                    train_set.append([source_ids, target_ids])
            line = fr1.readline()
        fr1.close()
        print('read over')
        
        rev_vocab = []
        rev_vocab.extend(data_utils._START_VOCAB)
        rev_vocab.extend(fr2.readlines())
        rev_vocab.extend(data_utils._END_VOCAB)
        fr2.close()
        rev_vocab = [i.strip() for i in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        print('write pickle start')
        fw_vocab = open('vocab','wb') 
        fw_rev_vocab = open('rev_vocab','wb') 
        fw_dev_set = open('dev_set','wb') 
        fw_train_set = open('train_set','wb') 
        pickle.dump(vocab, fw_vocab)
        pickle.dump(rev_vocab, fw_rev_vocab)
        pickle.dump(dev_set, fw_dev_set)
        pickle.dump(train_set, fw_train_set)
        fw_vocab.close()
        fw_rev_vocab.close()
        fw_dev_set.close()
        fw_train_set.close()
        print('write pickle done')
    return vocab, rev_vocab, dev_set, train_set


def create_model(gen_config):
    encoder = Encoder(gen_config.vocab_size, gen_config.emb_dim, gen_config.hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(gen_config.emb_dim, gen_config.hidden_size, gen_config.vocab_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=gen_config.lr)
    return seq2seq, optimizer

def getbatch(dataset, batch_size, maxlen):
    '''
    input:
    datase: [[source,target]...]
    output: 
    encoder_inputs: double list, padded input and reversee, (batch_size, maxlen)
    decoder_inputs: double list, padded output and add 'go' flag, (batch_size, maxlen), ('go' and 'end')
    source_inputs: double list, origin input, 
    target_inputs: double list, origin output, ('end')
    '''
    encoder_inputs = []
    decoder_inputs = []
    source_inputs = []
    target_inputs = []
    for i in range(batch_size):
        temp = random.choice(dataset)
        source = temp[0]
        source_inputs.append(source)
        target = temp[1]
        target_inputs.append(target)
        encoder_pad = [data_utils.PAD_ID] * (maxlen - len(source))
        encoder_inputs.append(list(reversed(source + encoder_pad)))
        decoder_pad = [data_utils.PAD_ID] * (maxlen - len(target) - 1)
        decoder_inputs.append([data_utils.GO_ID] + target + decoder_pad)
    return encoder_inputs, decoder_inputs, source_inputs, target_inputs
      
def train_with_reward(gen_config, seq2seq, optim_seq2seq, reward, train_query_neg, train_answer_gen):
    seq2seq.train()
    reward = reward*(gen_config.maxlen-1)
    src = torch.from_numpy(np.array(train_query_neg).T)
    trg = torch.from_numpy(np.array(train_answer_gen).T)
    src = Variable(src).cuda()
    trg = Variable(trg).cuda()
    optim_seq2seq.zero_grad()
    output = seq2seq(src, trg)
    loss = F.cross_entropy(output[1:].view(-1, gen_config.vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=data_utils.PAD_ID, reduce=False)
    print('train with reward loss is:', loss)
    # print('len reward:', len(reward))
    loss_reward = torch.dot(Variable(torch.Tensor(reward)).cuda(), loss)/len(reward)
    loss_reward.backward()
    clip_grad_norm(seq2seq.parameters(), gen_config.grad_clip)
    optim_seq2seq.step()
    return loss_reward.data[0]

def teacher_forcing(gen_config, seq2seq, optim_seq2seq, encoder_inputs, decoder_inputs):
    seq2seq.train()
    src = torch.from_numpy(np.array(encoder_inputs).T)
    trg = torch.from_numpy(np.array(decoder_inputs).T)
    src = Variable(src).cuda()
    trg = Variable(trg).cuda()
    optim_seq2seq.zero_grad()
    output = seq2seq(src, trg)
    loss = F.cross_entropy(output[1:].view(-1, gen_config.vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=data_utils.PAD_ID)
    print('teacher forcing loss is: ', loss)
    loss.backward()
    clip_grad_norm(seq2seq.parameters(), gen_config.grad_clip)
    optim_seq2seq.step()
    return loss.data[0]
    
def train(gen_config):
    print("begin pretrain gen model ...")
    vocab, rev_vocab, dev_set, train_set = prepare_data_new(gen_config)   
    #seqq2seq model
    seq2seq, optimizer = create_model(gen_config)
    seq2seq.train()
    pad = data_utils.PAD_ID
    total_loss = 0
    b = 0 
    while True:
        encoder_inputs, decoder_inputs, source_inputs, target_inputs = getbatch(train_set, gen_config.batch_size, gen_config.maxlen)
        b += 1
        src = torch.from_numpy(np.array(encoder_inputs).T)
        trg = torch.from_numpy(np.array(decoder_inputs).T)
        src = Variable(src).cuda()
        trg = Variable(trg).cuda()
        # src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = seq2seq(src, trg)
        loss = F.cross_entropy(output[1:].view(-1, gen_config.vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        loss.backward()
        clip_grad_norm(seq2seq.parameters(), gen_config.grad_clip)
        optimizer.step()
        
        total_loss += loss.data[0]
        print("single step loss", loss.data[0])
        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0
        if b % 1000 == 0:
            seq2seq.eval()
            total_val_loss = 0
            for i in range(100):
                val_encoder_inputs, val_decoder_inputs, val_source_inputs, val_target_inputs = getbatch(train_set, gen_config.batch_size, gen_config.maxlen)
                val_src = torch.from_numpy(np.array(val_encoder_inputs).T)
                val_trg = torch.from_numpy(np.array(val_decoder_inputs).T)
                val_src = Variable(val_src).cuda()
                val_trg = Variable(val_trg).cuda()
                # val_src, val_trg = val_src.cuda(), val_trg.cuda()
                val_output = seq2seq(val_src, val_trg)
                val_loss = F.cross_entropy(val_output[1:].view(-1, gen_config.vocab_size),
                               val_trg[1:].contiguous().view(-1),
                               ignore_index=pad)
                total_val_loss += val_loss.data[0]
            total_val_loss = total_val_loss/100.0
            print("[%d][val_loss:%5.2f][pp:%5.2f]" %
                  (b//1000, total_val_loss, math.exp(total_val_loss)))
            seq2seq.train()
            torch.save(seq2seq, './pre_seq2seq.pth')
            print("save model at step %d"%b)

def test_decoder(gen_config):
    # vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    # vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    fr_vocab = open('vocab','rb') 
    fr_rev_vocab = open('rev_vocab','rb') 
    vocab = pickle.load(fr_vocab)
    rev_vocab = pickle.load(fr_rev_vocab)
    fr_vocab.close()
    fr_rev_vocab.close()
    # seq2seq, optimizer = create_model(gen_config, vocab)
    seq2seq = torch.load('./pre_seq2seq.pth')
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        source = data_utils.sentence_to_token_ids(sentence, vocab) # list     
        encoder_pad = [data_utils.PAD_ID] * (gen_config.maxlen - len(source))
        encoder_inputs = [list(reversed(source + encoder_pad))]
        src = torch.from_numpy(np.array(encoder_inputs).T)
        src = Variable(src).cuda()
        probs, result = seq2seq.decode(src) # maxlen*1*vocab, a list
        ans = []
        for idx in result:
            if idx == data_utils.EOS_ID:
                break
            if idx != data_utils.PAD_ID:
                ans.append(rev_vocab[idx])
        print(" ".join(ans))
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

def gen_disc_data(gen_config):
    print("start gen disc data ...")
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config) 
    disc_train_query = open("train.query", "w")
    disc_train_answer = open("train.answer", "w")
    disc_train_gen = open("train.gen", "w")
    seq2seq  = torch.load('./pre_seq2seq.pth')
    num_steps = 0
    while num_steps<10000:
        encoder_inputs, decoder_inputs, source_inputs, target_inputs = getbatch(train_set, gen_config.batch_size, gen_config.maxlen)
        src = torch.from_numpy(np.array(encoder_inputs).T)
        trg = torch.from_numpy(np.array(decoder_inputs).T)
        src = Variable(src).cuda()
        trg = Variable(trg).cuda()
        probs, result = seq2seq.gen_data(src, trg)
        result = [list(i) for i in result]
        result = np.array(result).T.tolist()
        resps = []
        for each in result:
            temp = []
            for idx in each:
                if idx == data_utils.EOS_ID:
                    break
                if idx != data_utils.PAD_ID:
                    temp.append(idx)
            resps.append(temp)
        for query, answer, resp in zip(source_inputs, target_inputs, resps):
            answer_str = " ".join([str(rev_vocab[an]) for an in answer][:-1])
            disc_train_answer.write(answer_str)
            disc_train_answer.write("\n")

            query_str = " ".join([str(rev_vocab[qu]) for qu in query])
            disc_train_query.write(query_str)
            disc_train_query.write("\n")

            resp_str = " ".join([str(rev_vocab[output]) for output in resp])

            disc_train_gen.write(resp_str)
            disc_train_gen.write("\n")
        num_steps += 1
        if num_steps%5==0:
            print("generate disc data at %d step"%num_steps)
    disc_train_gen.close()
    disc_train_query.close()
    disc_train_answer.close()
    print("end gen disc data.")
        
            
    
    
