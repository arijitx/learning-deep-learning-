import load_data as ld
from models import encoder_decoder as m
import json
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

def module_hook(module, grad_input, grad_out):
	print('='*30)
	print('decoder module hook')
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
LEARNING_RATE=0.01
BATCH_SIZE=128
EMBED_SIZE=50
HIDDEN_SIZE=256
EPOCH=50
VOCAB_SIZE=len(word_2_id)
per_batch_print=5
train_loss=0

#define network components
print('> Creating Models ')
encoder=m.Encoder(VOCAB_SIZE,EMBED_SIZE,HIDDEN_SIZE,embs)
decoder=m.Decoder(EMBED_SIZE,HIDDEN_SIZE,VOCAB_SIZE,embs)
loss=nn.CrossEntropyLoss()

#define parameters to update
# params = list(encoder.gru.parameters()) + list(decoder.gru.parameters()) + list(decoder.out.parameters())
params=list(encoder.parameters())+list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

#define dataloader
print('> Loading Data')
d=ld.DataLoader('data/dev_sentence.txt','data/dev_question.txt','data/word2id.json',batch_size=BATCH_SIZE)
val_data=ld.DataLoader('data/dev_sentence.txt','data/dev_question.txt','data/word2id.json',batch_size=BATCH_SIZE)

print('> Starting Training')




for e in range(EPOCH):
	d.reset()
	for bi in d.n_batch:
		# zero accumalated grads of both encoder and decoder
		# encoder.zero_grad()
		# decoder.zero_grad()
		optimizer.zero_grad()
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

		# hn.register_hook(variable_hook)
		# now reverse the order of hn to suit the decoder datas lengths order
		# hn=hn.squeeze()
		# hn=hn[ienc]
		# hn=hn[idec]
		# hn=hn.unsqueeze(0)

		# decoder forward
		outputs,_,outputs_lens=decoder.forward(sb_train,hn,sbl)
		output_labels=outputs.max(2)[1]
		# print(output_labels.size())
		# compute loss
		sb_target_=sb_target
		sb_target=pack_padded_sequence(sb_target,sbl,batch_first=True)[0]
		outputs=pack_padded_sequence(outputs,outputs_lens,batch_first=True)[0]
		loss_=loss(outputs,sb_target)

		train_loss+=loss_.data[0]

		# if train_loss < 1. :
		# 	print(outputs.max(1)[0])
		# 	print(outputs.max(1)[1])
		# 	print('\n\n-------------------------\n\n')
		# 	print(sb_target)
		#backward prop and update params
		loss_.backward()
		optimizer.step()


		# print('> Epoch :',e+1,'Batch :',bi+1,'/',max(list(d.n_batch)),' Loss :',train_loss)
		# train_loss=0

		#validation
		if((bi+1)%per_batch_print==0) :
			#dump training pairs only for testing 
			temp_sentences=qb[ienc]
			temp_qbl=np.array(qbl)[ienc]
			temp_qbl=(temp_qbl[idec]).tolist()
			dump_text(output_labels,temp_sentences[idec],sbl,temp_qbl,bi,name='training_pairs_')

			val_loss=0
			val_data.n_batch=range(10)
			val_data.idx=0
			for vi in val_data.n_batch:
				vqb,_,vsb_target,vqbl,vsbl,ienc,idec=val_data.get_batch()

				vqb=Variable(torch.from_numpy(vqb),requires_grad=False).cuda(CUDA_DEVICE)
				vsb_target=Variable(torch.from_numpy(vsb_target)).cuda(CUDA_DEVICE)

				vsb_target=pack_padded_sequence(vsb_target,vsbl,batch_first=True)[0]
				output,hn=encoder(vqb,vqbl,[ienc,idec])

				output_label,output_value=decoder.infer(hn,word_2_id['EOS_TOKEN'],word_2_id['SOS_TOKEN'],max(vsbl),batch_size=BATCH_SIZE)
				output_value=pack_padded_sequence(output_value,vsbl,batch_first=True)[0]
				val_loss+=loss.forward(output_value,vsb_target).data[0]

				vqb=vqb[ienc]
				vqb=vqb[idec]

				vqbl=(np.array(vqbl)[ienc]).tolist()
				vqbl=(np.array(vqbl)[idec]).tolist()
				# print(output_label.size())
			dump_text(output_label,vqb,vsbl,vqbl,bi,name="var_dump_")
			val_loss=val_loss/max(list(val_data.n_batch))
			print('> Epoch :',e+1,'Batch :',bi+1,'/',max(list(d.n_batch)),' Loss :',train_loss/per_batch_print,' Val Loss :',val_loss)
			train_loss=0



