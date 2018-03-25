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
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

question_path='data/question.txt'
sentence_path='data/sentence.txt'
embs_path='data/embs.npy'
word_2_id_path='data/word2id.json'
unknown_words=set()
known_words=set()

CUDA_DEVICE=2



def module_hook(module, grad_input, grad_out):
	print('='*30)
	print('module hook')
	print('grad_out')
	print('='*30)
	print(grad_out)
	print('='*30)
	print('grad_input')
	print('='*30)
	print(grad_input)


def variable_hook(grad):
	print('='*30)
	print('variable hook')
	print('='*30)
	print('grad', grad)
	print('='*30)

class Encoder(nn.Module):
	def __init__(self,input_size,embed_size,hidden_size,embs):
		print('> Init Encoder')
		super(Encoder,self).__init__()
		self.hidden_size=hidden_size
		self.embedding=nn.Embedding(input_size,embed_size).cuda(CUDA_DEVICE)
		self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
		# self.embedding.weight.requires_grad=False
		self.gru=nn.GRU(embed_size,hidden_size,batch_first=True).cuda(CUDA_DEVICE)
		# self.gru.register_backward_hook(module_hook)
	
	def forward(self,input,lengths,idx):
		# https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106/14
		# lengths=torch.LongTensor(lengths).cuda(CUDA_DEVICE)
		# lengths, perm_index = lengths.sort(0, descending=True)
		# input = input[perm_index]
		# lengths=lengths.cpu().tolist()
		# print('encoder_input',input.size())
		embedded=self.embedding(input)
		packed_embed=pack_padded_sequence(embedded,lengths,batch_first=True)
		# print('encoder packed_input',packed_embed[0].size())
		output,hiddens=self.gru(packed_embed)
		# odx = perm_index.view(-1, 1).unsqueeze(0).expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
		idx1=Variable(torch.LongTensor(idx[0])).cuda(CUDA_DEVICE)
		idx2=Variable(torch.LongTensor(idx[1])).cuda(CUDA_DEVICE)
		
		idx1=idx1.unsqueeze(0).unsqueeze(2).expand(hiddens.size())
		idx2=idx2.unsqueeze(0).unsqueeze(2).expand(hiddens.size())

		hiddens = hiddens.gather(1, idx1)
		hiddens = hiddens.gather(1, idx2)
		# print('enocder hidden',hiddens.size())

		return output,hiddens

	
class Decoder(nn.Module):
	def __init__(self,embed_size,hidden_size,output_size,embs):
		print('> Init Decoder')
		super(Decoder,self).__init__()
		self.hidden_size=hidden_size
		self.embedding=nn.Embedding(output_size,embed_size).cuda(CUDA_DEVICE)
		self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
		# self.embedding.weight.requires_grad=False
		self.gru=nn.GRU(embed_size+hidden_size,hidden_size,batch_first=True).cuda(CUDA_DEVICE)
		self.out=nn.Linear(hidden_size,output_size).cuda(CUDA_DEVICE)
		# self.gru.register_backward_hook(module_hook)

	def forward(self,inputt,hidden,lengths):
		# print('decoder_hidden',hidden.size())
		output=self.embedding(inputt)
		h0=hidden
		k=output.size()[1]
		hidden=hidden.squeeze()
		hidden=hidden.unsqueeze(1)
		hidden=hidden.repeat(1,k,1)
		output=torch.cat((output,hidden),2)
		# print('decoder hidden_input_cat',output.size())
		packed_embed=pack_padded_sequence(output,lengths,batch_first=True)
		# print('decoder packed_hidden_input',packed_embed[0].size())
		output,hidden=self.gru(packed_embed,h0)
		output, output_lengths = pad_packed_sequence(output,batch_first=True)
		output=self.out(output)
		return output,hidden,output_lengths

	def infer(self,hidden,eos_token,sos_token,lens,batch_size=256):
		states=None
		output_label=[]
		output_value=[]

		start_token=Variable(torch.ones(batch_size,1).type(torch.LongTensor)*sos_token).cuda(CUDA_DEVICE)

		for i in range(lens):
			start_token=self.embedding(start_token)
			hidden=hidden.squeeze()
			hidden=hidden.unsqueeze(1)
			start_token=torch.cat((start_token,hidden),2)
			hiddens,states=self.gru(start_token,states)
			output=self.out(hiddens)
			output_value.append(output)
			label=output.max(2)[1]
			output_label.append(label)
			start_token=label

		return torch.cat(output_label,1),torch.cat(output_value,1)



