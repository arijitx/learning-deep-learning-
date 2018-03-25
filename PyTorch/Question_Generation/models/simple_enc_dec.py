import re
import numpy as np
from tqdm import tqdm
import mmap
import time
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

question_path='data/question.txt'
sentence_path='data/sentence.txt'
embs_path='data/embs.npy'
word_2_id_path='data/word2id.json'
unknown_words=set()
known_words=set()

CUDA_DEVICE=2


class Encoder(nn.Module):
	def __init__(self,input_size,hidden_size,embs):
		super(Encoder,self).__init__()
		self.hidden_size=hidden_size
		self.embedding=nn.Embedding(input_size,hidden_size).cuda(CUDA_DEVICE)
		self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
		self.embedding.weight.requires_grad=False
		self.gru=nn.GRU(hidden_size,hidden_size,batch_first=True).cuda(CUDA_DEVICE)
	
	def forward(self,input,lengths):
		embedded=self.embedding(input)
		packed_embed=pack_padded_sequence(embedded,lengths,batch_first=True)
		outputs,hiddens=self.gru(packed_embed)
		return outputs,hiddens

	
class Decoder(nn.Module):
	def __init__(self,hidden_size,output_size,embs):
		super(Decoder,self).__init__()
		self.hidden_size=hidden_size
		self.embedding=nn.Embedding(output_size,hidden_size).cuda(CUDA_DEVICE)
		self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
		self.embedding.weight.requires_grad=False
		self.gru=nn.GRU(hidden_size,hidden_size,batch_first=True).cuda(CUDA_DEVICE)
		self.out=nn.Linear(hidden_size,output_size).cuda(CUDA_DEVICE)

	def forward(self,inputt,hidden,lengths):
		output=self.embedding(inputt)
		packed_embed=pack_padded_sequence(output,lengths,batch_first=True)
		output,hidden=self.gru(packed_embed,hidden)
		output=self.out(output[0])
		return output,hidden	

	def infer(self,hidden,eos_token,sos_token,lens,batch_size=256):
		states=hidden
		output_label=[]
		output_value=[]

		start_token=Variable(torch.ones(batch_size,1).type(torch.LongTensor)*sos_token).cuda(CUDA_DEVICE)

		for i in range(lens):
			start_token=self.embedding(start_token)
			hiddens,states=self.gru(start_token,states)
			output=self.out(hiddens)
			output_value.append(output)
			label=output.max(2)[1]
			output_label.append(label)
			start_token=label

		return torch.cat(output_label,1),torch.cat(output_value,1)



