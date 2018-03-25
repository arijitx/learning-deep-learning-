import load_data as ld
from models import attentionRNN as m
import json
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

def dump_text(q,s,ql,sl,idx,name='qa_dump_'):
	# print(q.size())
	# print(s.size())
	id_2_word=dict(zip(word_2_id.values(),word_2_id.keys()))
	def word2id(x):
		return id_2_word[x]
	q=q.data.cpu().numpy().tolist()
	s=s.data.cpu().numpy().tolist()
	questions=[]
	sentences=[]
	for i in range(len(q)):
		questions.append(' '.join(list(map(word2id,q[i]))[:ql[i]]))

	for i in range(len(s)):
		sentences.append(' '.join(list(map(word2id,s[i]))[:sl[i]]))

	f=open('dump_qa/'+name+str(idx)+'.txt','w')
	f.write('\n\n'.join(sentences))
	f.write('\n------------------------\n')
	f.write('\n\n'.join(questions))
	f.close()

#load files
word_2_id=json.loads(open('data/word2id.json').read())
embs=np.load('data/embs.npy')

#define hyper params
CUDA_DEVICE=2
LEARNING_RATE=0.001
BATCH_SIZE=16
EMBED_SIZE=50
HIDDEN_SIZE=256
EPOCH=60
TEACHER_FORCING=0.3
VOCAB_SIZE=len(word_2_id)
per_batch_print=20
train_loss=0

#define network components
print('> Creating Models ')
encoder=m.EncoderRNN(VOCAB_SIZE,EMBED_SIZE,HIDDEN_SIZE,embs,n_layers=10)
decoder=m.AttnDecoderRNN(HIDDEN_SIZE,EMBED_SIZE,VOCAB_SIZE,embs,n_layers=10)
loss=nn.NLLLoss()

#define parameters to update
# params = list(encoder.gru.parameters()) + list(decoder.gru.parameters()) + list(decoder.out.parameters())
params=list(encoder.parameters())+list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

#define dataloader
print('> Loading Data')
d=ld.DataLoader('data/dev_sentence.txt','data/dev_question.txt','data/word2id.json',batch_size=BATCH_SIZE)
val_data=ld.DataLoader('data/sentence_small.txt','data/question_small.txt','data/word2id.json',batch_size=BATCH_SIZE)

print('> Starting Training')

for e in range(EPOCH):
	d.reset()
	train_loss=0
	for bi in d.n_batch:
		# zero accumalated grads of both encoder and decoder
		optimizer.zero_grad()
		encoder.train()
		decoder.train()

		# get batch with book keeping
		qb,sb_train,sb_target,qbl,sbl,ienc,idec=d.get_batch()
		# initialize Variables 
		# sb_train in format <sos> w1 w2 ... wn
		# sb_target in format w1 w2 ... wn <eos>
		qb=Variable(torch.from_numpy(qb)).cuda(CUDA_DEVICE)
		sb_train=Variable(torch.from_numpy(sb_train)).cuda(CUDA_DEVICE)
		sb_target=Variable(torch.from_numpy(sb_target)).cuda(CUDA_DEVICE)

		# encoder forward in batch
		hiddens,hn=encoder.forward(qb,qbl,[ienc,idec])

		# hiddens T*B*H
		# word_batch B
		# h n_layers*n_direction*B*H

		# decoder forward
		h0=hn
		h=h0
		outputs=[]
		outputs_labels=[]
		last_output=None

		for s in range(sb_train.size(1)):
			if torch.rand(1)[0] < TEACHER_FORCING or last_output is None:
				word_batch=sb_train[:,s].squeeze()
			else:
				word_batch=last_output
			# print('Training Word :',s)
			# print(word_batch)
			output,h=decoder.forward(word_batch,h,hiddens)
			last_output=output.max(1)[1]
			outputs_labels.append(last_output.unsqueeze(1))
			outputs.append(output.unsqueeze(1))

		outputs=torch.cat(outputs,1)
		outputs_labels=torch.cat(outputs_labels,1)
		outputs_packed=pack_padded_sequence(outputs,sbl,batch_first=True)[0]
		sb_target_packed=pack_padded_sequence(sb_target,sbl,batch_first=True)[0]
		loss_=loss(outputs_packed,sb_target_packed)
		train_loss+=loss_.data[0]
		loss_.backward()

		optimizer.step()
		# print('> Epoch :',e+1,'Batch :',bi+1,'/',max(list(d.n_batch)),' Loss :',train_loss)
		# train_loss=0

		#validation


		if((bi+1)%per_batch_print==0):
			#dump training pairs only for testing 
			# temp_sentences=qb[ienc]
			# temp_qbl=np.array(qbl)[ienc]
			# temp_qbl=(temp_qbl[idec]).tolist()
			# dump_text(output_labels,temp_sentences[idec],sbl,temp_qbl,bi,name='training_pairs_')
			encoder.eval()
			decoder.eval()
			val_loss=0
			val_data.n_batch=range(10)
			val_data.idx=0
			for vi in val_data.n_batch:
				vqb,_,vsb_target,vqbl,vsbl,ienc,idec=val_data.get_batch()
				vqb=Variable(torch.from_numpy(vqb),requires_grad=False).cuda(CUDA_DEVICE)
				vsb_target=Variable(torch.from_numpy(vsb_target)).cuda(CUDA_DEVICE)

				vsb_target=pack_padded_sequence(vsb_target,vsbl,batch_first=True)[0]
				output,hn=encoder(vqb,vqbl,[ienc,idec])

				output_label,output_value=decoder.infer(hn,output,word_2_id['EOS_TOKEN'],word_2_id['SOS_TOKEN'],max(vsbl),batch_size=BATCH_SIZE)
				output_value=pack_padded_sequence(output_value,vsbl,batch_first=True)[0]
				val_loss+=loss.forward(output_value,vsb_target).data[0]

				vqb=vqb[ienc]
				vqb=vqb[idec]

				vqbl=(np.array(vqbl)[ienc]).tolist()
				vqbl=(np.array(vqbl)[idec]).tolist()
				# dump_text(output_label,vqb,vsbl,vqbl,bi)
			val_loss=val_loss/(max(list(val_data.n_batch))+1.)
			print('> Epoch :',e+1,' Batch :',bi+1,'/',max(list(d.n_batch))+1,' Loss :',train_loss/per_batch_print,' Val Loss :',val_loss)
			f=open('output_log.txt','a')
			f.write('> Epoch :'+str(e+1)+' Batch :'+str(bi+1)+'/'+str(max(list(d.n_batch))+1)+' Loss :'+str(train_loss/per_batch_print)+' Val Loss :'+str(val_loss)+'\n')
			f.close()
			train_loss=0



