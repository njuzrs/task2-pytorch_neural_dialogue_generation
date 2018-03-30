import tensorflow as tf
import numpy as np
import torch 
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Hier_rnn_model(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(Hier_rnn_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                          dropout=dropout, bidirectional=False)
        self.lstm_context = nn.LSTM(hidden_size, hidden_size, num_layers,
                          dropout=dropout, bidirectional=False)
        self.linear = nn.Linear(hidden_size, 2)
        
        
    def forward(self, answer, query):
        # first lstm
        embedded_answer = self.embed(answer) # answer: [maxlen, batch], embedded:[maxlen,batch,emb_dims]
        outputs_answer, hidden_answer = self.lstm(embedded_answer)  #hidden :(h_n, c_n) h_n:[num_layers*num_direction, batch,hidden_size]
        last_hidden_answer = hidden_answer[1][-1] # [batch, hidden_size]
        
        embedded_query = self.embed(query)
        outputs_query, hidden_query = self.lstm(embedded_query)  #hidden :(h_n, c_n) h_n:[num_layers*num_direction, batch,hidden_size]
        last_hidden_query = hidden_query[1][-1] # [batch, hidden_size]
        
        # second lstm on sentence level
        context_input = torch.stack([last_hidden_answer, last_hidden_query]) #[2,batch,hidden_size]
        outputs_context, hidden_context = self.lstm_context(context_input)
        last_hidden_context = hidden_context[1][-1]
        
        # softmax
        output = self.linear(last_hidden_context)
        return output
