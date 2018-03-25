from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip
from six.moves import cPickle
import mmap
import re

import json
import numpy as np
from tqdm import tqdm #sexiest progresbar to ever exists

TRANSLATE = {
    "-lsb-" : "[",
    "-rsb-" : "]",
    "-lrb-" : "(",
    "-rrb-" : ")",
    "-lcb-" : "{",
    "-rcb-" : "}",
    "-LSB-" : "[",
    "-RSB-" : "]",
    "-LRB-" : "(",
    "-RRB-" : ")",
    "-LCB-" : "{",
    "-RCB-" : "}",
}


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    fp.close()
    return lines

def parse_args(description = "I am lazy"):
    import argparse
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("--embedding", type = str, default = "./data_qg/glove.6B.100d.txt", required = False)
    parser.add_argument("--question", type = str, required = False)
    parser.add_argument("--sentence", type = str, required = False)
    parser.add_argument("--output", type = str, required = False)
    parser.add_argument("--seed", type = int, default = 19941023)
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


def parse_words(filen_name):
    with open(filen_name,"r") as file:
        text = re.sub(r'\W+', ' ', file.read())
        text = re.sub(' +',' ',text)
        return set(text.split())

def create_words():
    args = parse_args()
    q_words=parse_words(args.question)
    s_words=parse_words(args.sentence)
    words=q_words.union(s_words)
    return list(words)

def main():
    args = parse_args()
    id2emb=[]
    word2id={}
    dimension = 50
    rejected_emb=0
    loaded_emb=0
    print('> Loading Glove Embedding File')
    with open(args.embedding, "r") as input_file:
        for line_x in tqdm(input_file, total=get_num_lines(args.embedding)):
            line = line_x.split()
            if len(line)==dimension+1:
                word2id[line[0]]=loaded_emb
                id2emb.append(np.asarray(list(map(float, line[-dimension: ]))))
                loaded_emb+=1
            else:
                rejected_emb+=1

    id2emb=np.array(id2emb)
    id2emb=np.reshape(id2emb,(id2emb.shape[0],id2emb[0].shape[0]))
    print(id2emb.shape)
    word2id['UNK_TOKEN']=loaded_emb
    word2id['PADD_TOKEN']=loaded_emb+1
    word2id['SOS_TOKEN']=loaded_emb+2
    word2id['EOS_TOKEN']=loaded_emb+3
    mean_embed=np.mean(id2emb,axis=0)
    mean_embed=np.expand_dims(mean_embed,0)

    padd_embed=np.zeros((1,dimension))
    sos_embed=np.random.normal(size=(1,dimension))
    eos_embed=np.random.normal(size=(1,dimension))
    id2emb=np.concatenate((id2emb , mean_embed,padd_embed,sos_embed,eos_embed))
    loaded_emb+=4

    print('> Loading Dataset . . ')
    words=create_words()
    new_word2id={}
    new_id2emb=[np.zeros(dimension)]
    emb_idx=1

    print('> Processing Words .. ')
    for i in tqdm(range(len(words))):
        w=words[i]
        if w in word2id:
            new_word2id[w]=emb_idx
            emb=id2emb[word2id[w]]
            new_id2emb.append(emb)
            emb_idx+=1

    print('> Found Embeddings :',emb_idx-1)
    print('> Embeddings not found :',len(words)-emb_idx+1)
    new_word2id['PADD_TOKEN']=0
    new_word2id['UNK_TOKEN']=emb_idx
    new_word2id['SOS_TOKEN']=emb_idx+1
    new_word2id['EOS_TOKEN']=emb_idx+2
    emb_idx+=2

    new_id2emb=np.array(new_id2emb)
    print(new_id2emb.shape)
    new_id2emb=np.concatenate((new_id2emb,mean_embed,sos_embed,eos_embed))
    print(new_id2emb.shape)
    new_id2emb=np.reshape(new_id2emb,(new_id2emb.shape[0],dimension))

    print('> Word2Id Len :',emb_idx+1)
    print('> Embedding Matrix Shape :',new_id2emb.shape)
    with open('data/word2id.json','w') as w2id_json:
        json.dump(new_word2id,w2id_json,indent=2)

    np.save(args.output,new_id2emb)
    print('> Rejected Embeddings :',rejected_emb)
    print('> Loaded Embeddings :',loaded_emb)


if __name__ == "__main__":
    main()
