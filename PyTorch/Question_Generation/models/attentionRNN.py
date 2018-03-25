import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

CUDA_DEVICE=2

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, embs, n_layers=1, dropout=0.):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size,embed_size).cuda(CUDA_DEVICE)
        self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
        # self.embedding.weight.requires_grad=False

        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True,dropout=self.dropout, bidirectional=True).cuda(CUDA_DEVICE)

    def forward(self, input_seqs, input_lengths, idx,hidden=None):
        # idx1=Variable(torch.LongTensor(idx[0])).cuda(CUDA_DEVICE)
        # idx2=Variable(torch.LongTensor(idx[1])).cuda(CUDA_DEVICE)

        input_lengths,order=Variable(torch.LongTensor(input_lengths)).cuda(CUDA_DEVICE).sort(descending=True)

        order=order.view(-1,1).expand(input_seqs.size())
        input_seqs=input_seqs.gather(0,order)

        embedded = self.embedding(input_seqs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,list(input_lengths.data),batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        # idx1=Variable(torch.LongTensor(idx[0])).cuda(CUDA_DEVICE)
        # idx1=idx1.view(-1,len(idx[0]),1).expand(outputs.size())
        
        # outputs=outputs.gather(1,idx1)

        idx1=Variable(torch.LongTensor(idx[0])).cuda(CUDA_DEVICE)
        idx2=Variable(torch.LongTensor(idx[1])).cuda(CUDA_DEVICE)

        idx1=idx1.unsqueeze(0).unsqueeze(2).expand(outputs.size())
        idx2=idx2.unsqueeze(0).unsqueeze(2).expand(outputs.size())


        outputs=outputs.gather(1, idx1)
        outputs=outputs.gather(1, idx2)

        hidden=hidden[0]+hidden[1]
        hidden=hidden.unsqueeze(0)

        idx1=Variable(torch.LongTensor(idx[0])).cuda(CUDA_DEVICE)
        idx2=Variable(torch.LongTensor(idx[1])).cuda(CUDA_DEVICE)
        
        idx1=idx1.unsqueeze(0).unsqueeze(2).expand(hidden.size())
        idx2=idx2.unsqueeze(0).unsqueeze(2).expand(hidden.size())


        hidden=hidden.gather(1, idx1)
        hidden=hidden.gather(1, idx2)

        hidden=hidden.repeat(self.n_layers,1,1)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size).cuda(CUDA_DEVICE)
        self.v = nn.Parameter(torch.rand(hidden_size)).cuda(CUDA_DEVICE)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies,dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size,embs,n_layers=1, dropout_p=0.):
        super(AttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size).cuda(CUDA_DEVICE)
        self.embedding.weight.data.copy_(torch.from_numpy(embs).cuda(CUDA_DEVICE))
        # self.embedding.weight.requires_grad=False
        self.dropout = nn.Dropout(dropout_p).cuda(CUDA_DEVICE)
        self.attn = Attn('concat', hidden_size).cuda(CUDA_DEVICE)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p).cuda(CUDA_DEVICE)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size).cuda(CUDA_DEVICE)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        # word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        # print('Train RNN input')
        # print(torch.sum(rnn_input))
        # print('Train Hidden :')
        # print(torch.sum(last_hidden))
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        context = context.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)),dim=1)
        # Return final output, hidden state

        return output, hidden

    def infer(self,last_hidden,encoder_outputs,eos_token,sos_token,lens,batch_size=256):
        states=None
        output_label=[]
        output_value=[]

        start_token=Variable(torch.ones(batch_size,1).type(torch.LongTensor)*sos_token).cuda(CUDA_DEVICE)
        for i in range(lens):
            # print("Validation Word :",i)
            # print(start_token)
            word_embedded=self.embedding(start_token).view(1,start_token.size(0),-1) # 1,B,V
            # word_embedded=self.dropout(word_embedded) # 1,B,V

            attn_weights=self.attn(last_hidden[-1],encoder_outputs)
            context=attn_weights.bmm(encoder_outputs.transpose(0,1)) # B,1,V
            context=context.transpose(0,1) # 1,B,H

            rnn_input=torch.cat((word_embedded,context),2) # 1,B,V+H
            # print("Validation RNN input :",i)
            # print(torch.sum(rnn_input))
            # print("Validation Hidden")
            # print(torch.sum(last_hidden))

            output,hidden=self.gru(rnn_input,last_hidden)


            output= output.squeeze(0)
            context=context.squeeze(0)

            output= F.log_softmax(self.out(torch.cat((output,context),1)),dim=1)

            start_token=output.max(1)[1]
            start_token=start_token.unsqueeze(1)

            output_value.append(output.unsqueeze(1))
            output_label.append(start_token)

            last_hidden=hidden

        return torch.cat(output_label,1),torch.cat(output_value,1)