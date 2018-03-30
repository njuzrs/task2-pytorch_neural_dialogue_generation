# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 15:29:38 2018

@author: zrssch
"""
import numpy as np
import sys
import time
import os
import gen.generator as gens
import utils.conf as conf
import disc.hier_disc as h_disc
import utils.data_utils as data_utils
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F

gen_config = conf.gen_config
disc_config = conf.disc_config
# evl_config = conf.disc_config

def gen_pre_train():
    gens.train(gen_config)
    
def gen_test():
    gens.test_decoder(gen_config)
    
# gen data for disc training
def gen_disc():
    gens.gen_disc_data(gen_config)
    
# pre train discriminator
def disc_pre_train():
    #discs.train_step(disc_config, evl_config)
    h_disc.hier_train(disc_config)
    
# prepare disc_data for discriminator and generator
def disc_train_data(seq2seq, vocab, source_inputs, target_inputs,
                    encoder_inputs, decoder_inputs, mc_search=False):
    train_query, train_answer, train_answer_gen = [], [], []
    maxlen = gen_config.maxlen
    pad = int(data_utils.PAD_ID)
    for query, answer in zip(source_inputs, target_inputs):
        query = query + [pad]*(maxlen-len(query))
        train_query.append(query)
        answer = answer[:-1] + [pad]*(maxlen-len(answer)+1) # remove eog
        train_answer.append(answer)
    train_labels = [1 for _ in source_inputs]
    
    # generate response
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
    assert len(resps) == len(train_query)
    for i, output in enumerate(resps):
        output = output + [pad] * (maxlen - len(output))
        if len(output) <= maxlen-2:
            output_gen = [data_utils.GO_ID] + output +[data_utils.EOS_ID] + [pad] * (maxlen - len(output)-2)
        else:
            output_gen = [data_utils.GO_ID] + output[:maxlen-2] +[data_utils.EOS_ID]
        train_query.append(train_query[i])
        train_answer.append(output)
        train_labels.append(0)
        train_answer_gen.append(output_gen)
    return train_query, train_answer, train_labels, train_answer_gen

# Adversarial Learning for Neural Dialogue Generation
def al_train():
    vocab, rev_vocab, dev_set, train_set = gens.prepare_data(gen_config)
    
    seq2seq = torch.load('pre_seq2seq.pth')   
    optim_seq2seq = optim.Adam(seq2seq.parameters(), lr=gen_config.lr)
    hrnn = torch.load('pre_hrnn.pth')  
    optim_hrnn = optim.Adam(hrnn.parameters(), lr=disc_config.lr)
    # hrnn, optim_hrnn = h_disc.create_model(disc_config)
    # seq2seq, optim_seq2seq = gens.create_model(gen_config)
    
    current_step = 0
    while True:
        current_step += 1
        start_time = time.time()
        print("==================Update Discriminator: %d=====================" % current_step)
        for i in range(disc_config.disc_steps):
            # 1.Sample (X,Y) from real disc_data
            encoder_inputs, decoder_inputs, source_inputs, target_inputs = gens.getbatch(train_set, gen_config.batch_size, gen_config.maxlen)
            
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query, train_answer, train_labels, train_answer_gen = disc_train_data(seq2seq, vocab, source_inputs, target_inputs,
                                                            encoder_inputs, decoder_inputs, mc_search=False)
            print("==============================mc_search: False===================================")
            if current_step % 200 == 0:
                print("train_query: ", len(train_query))
                print("train_answer: ", len(train_answer))
                print("train_labels: ", len(train_labels))
                for i in xrange(len(train_query)):
                    print("lable: ", train_labels[i])
                    print("train_answer_sentence: ", train_answer[i])
                    print(" ".join([rev_vocab[output] for output in train_answer[i]]))
                    
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            step_loss = h_disc.disc_step(hrnn, optim_hrnn, disc_config, train_query, train_answer, train_labels)
            print("update discriminator loss is:", step_loss)
        
        for i in range(gen_config.gen_steps):
            print("==================Update Generator: %d=========================" % current_step)
            # 1.Sample (X,Y) from real disc_data
            encoder_inputs, decoder_inputs, source_inputs, target_inputs = gens.getbatch(train_set, gen_config.batch_size, gen_config.maxlen)
            
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query, train_answer, train_labels, train_answer_gen = disc_train_data(seq2seq, vocab, source_inputs, target_inputs,
                                                            encoder_inputs, decoder_inputs, mc_search=False)
            train_query_neg = []
            train_answer_neg = []
            train_labels_neg = []
            train_query_pos = []
            train_answer_pos = []
            train_labels_pos = []
            for j in range(len(train_labels)):
                if train_labels[j] == 0:
                    train_query_neg.append(train_query[j])
                    train_answer_neg.append(train_answer[j])
                    train_labels_neg.append(0)
                else:
                    train_query_pos.append(train_query[j])
                    train_answer_pos.append(train_answer[j])
                    train_labels_pos.append(1)
            
            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward = h_disc.disc_reward_step(hrnn, train_query_neg, train_answer_neg)
            # 4.update G on (X, ^Y) using reward r
            loss_reward = gens.train_with_reward(gen_config, seq2seq, optim_seq2seq, reward, train_query_neg, train_answer_gen)
            # 5.Teacher-Forcing: update G on (X, Y)
            loss = gens.teacher_forcing(gen_config, seq2seq, optim_seq2seq, encoder_inputs, decoder_inputs)
            print("update generate loss, reward is %f, loss_reward is %f, loss is %f"%(np.mean(reward), loss_reward, loss))
        end_time = time.time()
        print("step %d spend time: %f"%(current_step, end_time-start_time))
        
        if current_step%1000 == 0:
            torch.save(seq2seq, './seq2seq.pth')
            torch.save(hrnn, './hrnn.pth')
            
            
if __name__ == "__main__":
    # gen_pre_train()
    # gen_test()
    # step_2 gen training data for disc
    # gen_disc()
    
    # step_3 training disc model
    # disc_pre_train()
    
    # step_4 training al model
    al_train()
