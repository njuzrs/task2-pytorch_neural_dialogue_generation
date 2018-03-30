import numpy as np
import os
import time
import datetime
import random
import utils.data_utils as data_utils
from hier_rnn_model import Hier_rnn_model
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch import nn
import math
import pickle

sys.path.append("../utils")

def hier_get_batch(config, query_set, answer_set, gen_set):
    max_set = len(query_set) - 1
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = batch_size / 2
    for _ in range(half_size):
        index = random.randint(0, max_set)
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        train_labels.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        train_labels.append(0)
    return train_query, train_answer, train_labels

def create_model(config):
    model = Hier_rnn_model(config.vocab_size, config.embed_dim, config.embed_dim, config.num_layers).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    return model, optimizer

def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = []
    answer_set = []
    gen_set = []
    query_file = open(query_path, 'r')
    answer_file = open(answer_path, 'r')
    gen_file = open(gen_path, 'r')
    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
    counter = 0
    while query and answer and gen:
        counter += 1
        if counter % 100000 == 0:
            print("  reading disc_data line %d" % counter)
        query = [int(id) for id in query.strip().split()]
        answer = [int(id) for id in answer.strip().split()]
        gen = [int(id) for id in gen.strip().split()]
        if max(len(query), len(answer), len(gen))<=config.maxlen:
            query = query + [data_utils.PAD_ID]*(config.maxlen-len(query))
            query_set.append(query)
            answer = answer + [data_utils.PAD_ID]*(config.maxlen-len(answer))
            answer_set.append(answer)
            gen = gen + [data_utils.PAD_ID]*(config.maxlen-len(gen))
            gen_set.append(gen)
        query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
    query_file.close()
    answer_file.close()
    gen_file.close()
    return query_set, answer_set, gen_set

def prepare_data(config):
    '''
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    '''
    fr_vocab = open('vocab','rb') 
    fr_rev_vocab = open('rev_vocab','rb') 
    vocab = pickle.load(fr_vocab)
    rev_vocab = pickle.load(fr_rev_vocab)
    fr_vocab.close()
    fr_rev_vocab.close()
    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
        data_utils.hier_prepare_disc_data(config.train_dir, vocab, config.vocab_size)
    query_set, answer_set, gen_set = hier_read_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set
    
def hier_train(config_disc):
    print("begin pre disc training ...")
    query_set, answer_set, gen_set = prepare_data(config_disc)
    hrnn, optimizer = create_model(config_disc)
    hrnn.train()
    total_loss = 0
    b = 0
    while True:
        train_query, train_answer, train_labels = hier_get_batch(config_disc, query_set, answer_set, gen_set)
        b += 1
        train_query = torch.from_numpy(np.array(train_query).T)
        train_answer = torch.from_numpy(np.array(train_answer).T)
        train_labels = torch.from_numpy(np.array(train_labels))
        train_query = Variable(train_query).cuda()
        train_answer = Variable(train_answer).cuda()
        train_labels = Variable(train_labels).cuda()
        
        optimizer.zero_grad()
        output = hrnn(train_query, train_answer)
        loss = F.cross_entropy(output, train_labels)
        loss.backward()
        clip_grad_norm(hrnn.parameters(), config_disc.grad_clip)
        optimizer.step()
        
        total_loss += loss.data[0]
        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0
        
        if b%1000 == 0:
            hrnn.eval()
            total_val_loss = 0
            for i in range(100):
                val_query, val_answer, val_labels = hier_get_batch(config_disc, query_set, answer_set, gen_set)
                val_query = torch.from_numpy(np.array(val_query).T)
                val_answer = torch.from_numpy(np.array(val_answer).T)
                val_labels = torch.from_numpy(np.array(val_labels))
                val_query = Variable(val_query).cuda()
                val_answer = Variable(val_answer).cuda()
                val_labels = Variable(val_labels).cuda()
                val_output = hrnn(val_query, val_answer)
                val_loss = F.cross_entropy(val_output, val_labels)
                total_val_loss += val_loss.data[0]
            total_val_loss = total_val_loss/100.0
            print("[%d][val_loss:%5.2f][pp:%5.2f]" %
                  (b//1000, total_val_loss, math.exp(total_val_loss)))
            hrnn.train()
            torch.save(hrnn, './pre_hrnn.pth')
            print("save pre hrnn model at step %d"%b)
                
def disc_step(hrnn, optim_hrnn, config_disc, train_query, train_answer, train_labels):
    print("begin disc step train")
    hrnn.train()
    train_query = torch.from_numpy(np.array(train_query).T)
    train_answer = torch.from_numpy(np.array(train_answer).T)
    train_labels = torch.from_numpy(np.array(train_labels))
    train_query = Variable(train_query).cuda()
    train_answer = Variable(train_answer).cuda()
    train_labels = Variable(train_labels).cuda()
    
    optim_hrnn.zero_grad()
    output = hrnn(train_query, train_answer)
    loss = F.cross_entropy(output, train_labels)
    loss.backward()
    clip_grad_norm(hrnn.parameters(), config_disc.grad_clip)
    optim_hrnn.step()
    return loss.data[0]
    
def disc_reward_step(hrnn, train_query_neg, train_answer_neg): 
    print("get disc reward step")
    hrnn.train()
    train_query_neg = torch.from_numpy(np.array(train_query_neg).T)
    train_answer_neg = torch.from_numpy(np.array(train_answer_neg).T)
    train_query_neg = Variable(train_query_neg).cuda()
    train_answer_neg = Variable(train_answer_neg).cuda()
    output = hrnn(train_query_neg, train_answer_neg)
    logit = nn.Softmax()(output)
    reward = list(logit.data[:,1])
    return reward
    
    
    

    





