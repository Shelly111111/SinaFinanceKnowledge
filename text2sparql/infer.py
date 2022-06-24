# -*- coding: gbk -*-

import io
import os

from functools import partial

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.data import Vocab, Pad
from paddlenlp.metrics import Perplexity
from paddlenlp.datasets import load_dataset

import jieba
import json


pad_id = 2
bos_id = 1
eos_id = 2
data_path='data.txt'
data_dic_path = 'data_dic.txt'

def get_id(vocab_dict, token):
    return vocab_dict[token]

with open(data_path,'r',encoding='utf-8') as fp:
    lines=fp.readlines()
    vocabs=[]
    quers=[]
    asws=[]
    for line in lines:
        line = line.lower() #全部转为小写
        data=line.split('\t')
        quer=jieba.lcut(data[0].strip(), cut_all=False)
        asw=data[1].strip().split(' ')
        vocabs.extend(quer)
        vocabs.extend(asw)
        quers.append(quer)
        asws.append(asw)
    vocabs=list(set(vocabs))
with open(data_dic_path,'r',encoding='utf-8') as fp:
    vocab_dict=json.loads(fp.readline())
print(vocab_dict)

def read(data_path,quers,asws,vocab_dict):
    for q,a in zip(quers,asws):
        quer = [bos_id]+[get_id(vocab_dict, v) for v in q]+[eos_id]
        asw = [bos_id]+[get_id(vocab_dict, v) for v in a]+[eos_id]
        #train_ds.append(([bos_id]+quer+[eos_id],[bos_id]+asw+[eos_id]))
        yield (quer, asw)
train_ds = load_dataset(read, data_path=data_path, quers=quers, asws=asws, vocab_dict=vocab_dict, lazy=False)
test_ds = train_ds
vocab_size = len(vocabs)+3
print(vocab_dict)
print(vocab_size)
trg_idx2word = {k:v for v,k in vocab_dict.items()}

def create_data_loader(dataset):
    data_loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=None,
        batch_size = batch_size,
        collate_fn=partial(prepare_input, pad_id=pad_id))
    return data_loader

def prepare_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id, ret_length=True)([inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)([inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype(paddle.get_default_dtype())
    #print(src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask)
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask

device = "gpu" # or cpu
device = paddle.set_device(device)

batch_size = 6
num_layers = 2
dropout = 0.2
hidden_size =256
max_grad_norm = 5.0
learning_rate = 0.001
max_epoch = 300
model_path = './couplet_models'
log_freq = 2000

# Define dataloader
test_loader = create_data_loader(test_ds)

class Seq2SeqEncoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0.)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)
        
        # encoder_output [128, 18, 256]  [batch_size, time_steps, hidden_size]
        # encoder_state (tuple) - 最终状态,一个包含h和c的元组。 [2, 128, 256] [2, 128, 256] [num_layers * num_directions, batch_size, hidden_size]
        return encoder_output, encoder_state

class AttentionLayer(nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        encoder_output = self.input_proj(encoder_output)
        attn_scores = paddle.matmul(
            paddle.unsqueeze(hidden, [1]), encoder_output, transpose_y=True)
        # print('attention score', attn_scores.shape) #[128, 1, 18]

        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)

        attn_scores = F.softmax(attn_scores)
        attn_out = paddle.squeeze(
            paddle.matmul(attn_scores, encoder_output), [1])
        # print('1 attn_out', attn_out.shape) #[128, 256]

        attn_out = paddle.concat([attn_out, hidden], 1)
        # print('2 attn_out', attn_out.shape) #[128, 512]

        attn_out = self.output_proj(attn_out)
        # print('3 attn_out', attn_out.shape) #[128, 256]
        return attn_out

class Seq2SeqDecoderCell(nn.RNNCellBase):
    def __init__(self, num_layers, input_size, hidden_size):
        super(Seq2SeqDecoderCell, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm_cells = nn.LayerList([
            nn.LSTMCell(
                input_size=input_size + hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size) for i in range(num_layers)
        ])

        self.attention_layer = AttentionLayer(hidden_size)
    
    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            step_input = self.dropout(out)
            new_lstm_states.append(new_lstm_state)
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]

class Seq2SeqDecoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.lstm_attention = nn.RNN(
            Seq2SeqDecoderCell(num_layers, embed_dim, hidden_size))
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, trg, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(trg)

        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)

        return predict

class Seq2SeqAttnModel(nn.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 eos_id=1):
        super(Seq2SeqAttnModel, self).__init__()
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_size,
                                      num_layers)

    def forward(self, src, src_length, trg):
        # encoder_output 各时刻的输出h
        # encoder_final_state 最后时刻的输出h，和记忆信号c
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        # print('encoder_output shape', encoder_output.shape)  #  [128, 18, 256]  [batch_size,time_steps,hidden_size]
        # print('encoder_final_states shape', encoder_final_state[0].shape, encoder_final_state[1].shape) #[2, 128, 256] [2, 128, 256] [num_lauers * num_directions, batch_size, hidden_size]

        # Transfer shape of encoder_final_states to [num_layers, 2, batch_size, hidden_size]？？？
        encoder_final_states = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]
        # print('encoder_final_states shape', encoder_final_states[0][0].shape, encoder_final_states[0][1].shape) #[128, 256] [128, 256]


        # Construct decoder initial states: use input_feed and the shape is
        # [[h,c] * num_layers, input_feed], consistent with Seq2SeqDecoderCell.states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]

        # Build attention mask to avoid paying attention on padddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        # print ('src_mask shape', src_mask.shape)  #[128, 18]
        # print(src_mask[0, :])

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        # print ('encoder_padding_mask', encoder_padding_mask.shape)  #[128, 18]
        # print(encoder_padding_mask[0, :])

        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
        # print('encoder_padding_mask', encoder_padding_mask.shape)  #[128, 1, 18]

        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        # print('predict', predict.shape)   #[128, 17, 7931]

        return predict


class Seq2SeqAttnInferModel(Seq2SeqAttnModel):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers
        super(Seq2SeqAttnInferModel, self).__init__(
            vocab_size, embed_dim, hidden_size, num_layers, eos_id)

        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        encoder_final_state = [
            (encoder_final_state[0][i], encoder_final_state[1][i])
            for i in range(self.num_layers)
        ]

        # Initial decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # Build attention mask to avoid paying attention on paddings
        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())

        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return seq_output

def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq

beam_size = 10
model = paddle.Model(
    Seq2SeqAttnInferModel(
        vocab_size,
        hidden_size,
        hidden_size,
        num_layers,
        bos_id=bos_id,
        eos_id=eos_id,
        beam_size=beam_size,
        max_out_len=256))

model.prepare()

model.load('model/final')

idx = 0
trg_idx2word[1]='<start>'
trg_idx2word[2]='<end>'
for data in test_loader():
    inputs = data[:2]
    print(inputs)
    finished_seq = model.predict_batch(inputs=list(inputs))[0]
    finished_seq = finished_seq[:, :, np.newaxis] if len(
        finished_seq.shape) == 2 else finished_seq
    finished_seq = np.transpose(finished_seq, [0, 2, 1])
    for ins in finished_seq:
        for beam in ins:
            id_list = post_process_seq(beam, bos_id, eos_id)
            word_list_f = [trg_idx2word[id] for id in test_ds[idx][0]][1:-1]
            word_list_s = [trg_idx2word[id] for id in id_list]
            sequence = "问: "+"".join(word_list_f)+"\tsparql: "+" ".join(word_list_s).replace('sct:haschineselabel','sct:hasChineseLabel').replace('sct:id','sct:ID').replace('zg:sinafinance','zg:SinaFinance') + "\n"
            print(sequence)
            idx += 1
            break