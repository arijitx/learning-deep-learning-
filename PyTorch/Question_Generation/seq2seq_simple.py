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

	def init_hidden(self):
		result=Variable(torch.zeros(1,1,self.hidden_size))
		return result.cuda()
	
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
		output=F.relu(output)
		packed_embed=pack_padded_sequence(output,lengths,batch_first=True)
		output,hidden=self.gru(packed_embed,hidden)
		output=self.out(output[0])
		return output,hidden	

class Seq2Seq():
	def __init__(self,encoder,decoder,criterion=None,enc_opt=None,dec_opt=None):
		self.encoder=encoder
		self.decoder=decoder
		self.enc_opt=enc_opt or optim.SGD(self.encoder.parameters(),lr=0.01)
		self.dec_opt=dec_opt or optim.SGD(self.decoder.parameters(),lr=0.01)
		self.criterion=criterion or nn.NLLLoss()
		self.loss=[]
		self.print_every=200

	def train(self,sentence,question,max_iter=None):
		As=sentence
		Qs=question
		max_iter = max_iter or As.shape[0]
		t = tqdm(range(max_iter))
		loss=0
		epoch=2
		for e in range(epoch):
			t.set_description("Epoch ",e)
			t.refresh()
			for i in t:
				loss += self.one_iteration(As[i],Qs[i])
				if((i+1)%self.print_every==0):
					self.loss.append(loss/self.print_every)
					print('> Loss :',loss/self.print_every)
					loss=0


	def predict(self,sentence):
		print('> Loading Prediction Text')
		sentence=prepare_data(sentence)
		result=''
		print('> Predicting ')
		for i in tqdm(range(sentence.shape[0])):
			sent=Variable(torch.from_numpy(sentence[i])).view(-1,1).cuda()
			s_len=sent.size()[0]
			sos=Variable(torch.LongTensor([input_size-1])).view(-1,1).cuda()
			eh=self.encoder.init_hidden()
			for j in range(s_len):
				eo,eh=self.encoder(sent[j],eh)

			dh=eh
			do=sos
			for k in range(100):
				do,dh=self.decoder(do,dh)
				topv, topi = do.data.topk(1)
				ni = topi[0][0]
				do=Variable(torch.LongTensor([[ni]])).cuda()
				if ni==eos_token:
					break
				result+=id_2_word[ni]+' '
			result+='?\n'
		f=open('generated_questions.txt','w')
		f.write(result)
		f.close()
		print('> Done Prediction')


	def one_iteration(self,sentence,question):
		encoder_hidden=self.encoder.init_hidden()
		
		self.enc_opt.zero_grad()
		self.dec_opt.zero_grad()
		
		sentence=Variable(torch.from_numpy(sentence)).view(-1,1).cuda()
		question=Variable(torch.from_numpy(question)).view(-1,1).cuda()
		
		s_len=sentence.size()[0]
		q_len=question.size()[0]
		
		loss=0
		for ei in range(s_len):
			encoder_output,encoder_hidden=self.encoder.forward(sentence[ei],encoder_hidden)
		sos=Variable(torch.LongTensor([input_size-1])).view(-1,1).cuda()
		decoder_hidden=encoder_hidden
		decoder_output=sos
		for di in range(q_len):
			decoder_output,decoder_hidden=self.decoder(decoder_output,decoder_hidden)
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]
			loss+=self.criterion(decoder_output,question[di])
			decoder_output=Variable(torch.LongTensor([[ni]])).cuda()
			if ni==eos_token:
				break
			
		loss.backward()
		self.enc_opt.step()
		self.dec_opt.step()
		return loss.data[0]/q_len


# print('> Preparing Vocabulary')
# word_2_id=json.loads(open(word_2_id_path).read())
# embeddings=np.load(embs_path)
# print('> Vocab Len : ',len(word_2_id))
# print('> Embs Shape : ',embeddings.shape)

# vocab_size=len(word_2_id)
# print('> Loading Questions ')
# Qs=prepare_data(question_path)
# print('> Loading Sentences ')
# As=prepare_data(sentence_path)

# print('> Unknow Words',len(unknown_words))
# print('> Known Words',len(known_words))
# # print('> Starting Training ')
# # input_size=vocab_size
# # hidden_size=300
# # enc=Encoder(input_size,hidden_size,embeddings).cuda()
# # dec=Decoder(hidden_size,input_size,embeddings).cuda()

# # s2s=Seq2Seq(enc,dec)
# # s2s.train(As,Qs)
# # s2s.predict('data/test_sentence.txt')





