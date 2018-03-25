import numpy as np 
import json
from tqdm import tqdm
import mmap
import re
import torch
import math

class DataLoader():
	def __init__(self,q_path,s_path,w2id_path,batch_size=64):
		self.word_2_id=json.loads(open(w2id_path).read())
		self.known_words=set()
		self.unknown_words=set()
		self.question=self.prepare_data(q_path)
		self.sentence=self.prepare_data(s_path)
		self.idx=0
		self.batch_size=batch_size
		self.n_batch=range(math.floor(self.question.shape[0]/batch_size))

	def reset(self):
		self.idx=0
		p = np.random.permutation(self.question.shape[0])
		self.question=self.question[p]
		self.sentence=self.sentence[p]

	def get_batch(self):
		qbatch=self.question[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
		sbatch=self.sentence[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
		q_max_len=0
		s_max_len=0
		for i in range(qbatch.shape[0]):
			q_max_len=max(qbatch[i].shape[0],q_max_len)
		for i in range(sbatch.shape[0]):
			s_max_len=max(sbatch[i].shape[0],s_max_len)

		new_q_batch=[]
		new_s_batch_train=[]
		new_s_batch_target=[]
		q_batch_len=[]
		s_batch_len=[]
		for i in range(qbatch.shape[0]):
			new_q_batch.append(np.concatenate((qbatch[i],np.ones((q_max_len-qbatch[i].shape[0]))*self.word_2_id['PADD_TOKEN'])).astype('int'))
			q_batch_len.append(qbatch[i].shape[0])

		for i in range(sbatch.shape[0]):
			new_s_batch_train.append(np.concatenate(([self.word_2_id['SOS_TOKEN']],sbatch[i],np.ones((s_max_len-sbatch[i].shape[0]))*self.word_2_id['PADD_TOKEN'])).astype('int'))
			new_s_batch_target.append(np.concatenate((sbatch[i],[self.word_2_id['EOS_TOKEN']],np.ones((s_max_len-sbatch[i].shape[0]))*self.word_2_id['PADD_TOKEN'])).astype('int'))
			s_batch_len.append(sbatch[i].shape[0]+1)

		
		new_q_batch=np.array(new_q_batch).reshape((len(new_q_batch),new_q_batch[0].shape[0]))
		new_s_batch_train=np.array(new_s_batch_train)
		new_s_batch_target=np.array(new_s_batch_target)

		q_batch_len=list(map(int,q_batch_len))
		s_batch_len=list(map(int,s_batch_len))

		idx=np.array(q_batch_len).argsort()[::-1]
		k=dict(enumerate(idx.tolist()))
		inv_idx_encoder=list(dict(zip(k.values(),k.keys())).values())

		q_batch_len=(np.array(q_batch_len)[idx.tolist()]).tolist()
		new_q_batch=new_q_batch[idx]

		idx=np.array(s_batch_len).argsort()[::-1]
		k=dict(enumerate(idx.tolist()))
		inv_idx_decoder=list(dict(zip(k.values(),k.keys())).values())

		s_batch_len=(np.array(s_batch_len)[idx.tolist()]).tolist()
		new_s_batch_train=new_s_batch_train[idx]
		new_s_batch_target=new_s_batch_target[idx]

		new_q_batch=new_q_batch.reshape((new_q_batch.shape[1],new_q_batch.shape[0]))
		new_s_batch_train=new_s_batch_train.reshape((new_s_batch_train.shape[1],new_s_batch_train.shape[0]))

		# q_batch_len=(np.array(q_batch_len)[idx.tolist()]).tolist()
		# new_q_batch=new_q_batch[idx]
		self.idx+=1
		return new_q_batch,new_s_batch_train,new_s_batch_target,q_batch_len,s_batch_len,inv_idx_encoder,idx.tolist()


	def get_num_lines(self,file_path):
		fp = open(file_path, "r+")
		buf = mmap.mmap(fp.fileno(), 0)
		lines = 0
		while buf.readline():
			lines += 1
		fp.close()
		return lines

	def str_2_id(self,x):
		if x in self.word_2_id:
			self.known_words.add(x)
			return self.word_2_id[x]
		else:
			self.unknown_words.add(x)
			return self.word_2_id['UNK_TOKEN']

	def prepare_data(self,text):
		X=[]
		with open(text,'r') as f:
			for line in tqdm(f,total=self.get_num_lines(text)):
				line = re.sub(r'\W+', ' ', line)
				line = re.sub(r' +',' ',line)
				X.append(np.array(list(map(self.str_2_id,line.split()))))
		return np.array(X)

# d=DataLoader('data/question.txt','data/sentence.txt','data/word2id.json')
# qb,sb_train,sb_test,qbl,sbl=d.get_batch()

# print(qb.shape)
# print(sb_train.shape)
# print(sb_test.shape)