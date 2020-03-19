import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
import numpy as np
from pandas.io.json import json_normalize

opt1 = json.load(open('./save_model/vatex_msrvtt_res152_i3d/opt_info.json', 'r'))
opt2 = json.load(open('./save_model/vatex_msrvtt_res152/opt_info.json', 'r'))
opt3 = json.load(open('./save_model/vatex_msrvtt_msvd_res152_i3d/opt_info.json', 'r'))

dataset1 = VideoDataset(opt1, "test")
dataset2 = VideoDataset(opt2, "test")
dataset3 = VideoDataset(opt3, "test")

opt1["vocab_size"] = 27170
opt1["seq_length"] = 28
opt2["vocab_size"] = 27170
opt2["seq_length"] = 28
opt3["vocab_size"] = 28620
opt3["seq_length"] = 28

encoder = EncoderRNN(opt1["dim_vid"], opt1["dim_hidden"], bidirectional=bool(opt1["bidirectional"]),
                             input_dropout_p=opt1["input_dropout_p"], rnn_dropout_p=opt1["rnn_dropout_p"])
decoder = DecoderRNN(opt1["vocab_size"], opt1["max_len"], opt1["dim_hidden"], opt1["dim_word"],
                             input_dropout_p=opt1["input_dropout_p"],
                             rnn_dropout_p=opt1["rnn_dropout_p"], bidirectional=bool(opt1["bidirectional"]))
model1 = S2VTAttModel(encoder, decoder).cuda()
model1.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_res152_i3d/model_320.pth').items()})

encoder = EncoderRNN(opt1["dim_vid"], opt1["dim_hidden"], bidirectional=bool(opt1["bidirectional"]),
                             input_dropout_p=opt1["input_dropout_p"], rnn_dropout_p=opt1["rnn_dropout_p"])
decoder = DecoderRNN(opt1["vocab_size"], opt1["max_len"], opt1["dim_hidden"], opt1["dim_word"],
                             input_dropout_p=opt1["input_dropout_p"],
                             rnn_dropout_p=opt1["rnn_dropout_p"], bidirectional=bool(opt1["bidirectional"]))
model2 = S2VTAttModel(encoder, decoder).cuda()
model2.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_res152_i3d/model_300.pth').items()})

encoder = EncoderRNN(opt2["dim_vid"], opt2["dim_hidden"], bidirectional=bool(opt2["bidirectional"]),
                             input_dropout_p=opt2["input_dropout_p"], rnn_dropout_p=opt2["rnn_dropout_p"])
decoder = DecoderRNN(opt2["vocab_size"], opt2["max_len"], opt2["dim_hidden"], opt2["dim_word"],
                             input_dropout_p=opt2["input_dropout_p"],
                             rnn_dropout_p=opt2["rnn_dropout_p"], bidirectional=bool(opt2["bidirectional"]))
model3 = S2VTAttModel(encoder, decoder).cuda()
model3.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_res152/model_320.pth').items()})

encoder = EncoderRNN(opt2["dim_vid"], opt2["dim_hidden"], bidirectional=bool(opt2["bidirectional"]),
                             input_dropout_p=opt2["input_dropout_p"], rnn_dropout_p=opt2["rnn_dropout_p"])
decoder = DecoderRNN(opt2["vocab_size"], opt2["max_len"], opt2["dim_hidden"], opt2["dim_word"],
                             input_dropout_p=opt2["input_dropout_p"],
                             rnn_dropout_p=opt2["rnn_dropout_p"], bidirectional=bool(opt2["bidirectional"]))
model4 = S2VTAttModel(encoder, decoder).cuda()
model4.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_res152/model_280.pth').items()})

encoder = EncoderRNN(opt3["dim_vid"], opt3["dim_hidden"], bidirectional=bool(opt3["bidirectional"]),
                             input_dropout_p=opt3["input_dropout_p"], rnn_dropout_p=opt3["rnn_dropout_p"])
decoder = DecoderRNN(opt3["vocab_size"], opt3["max_len"], opt3["dim_hidden"], opt3["dim_word"],
                             input_dropout_p=opt3["input_dropout_p"],
                             rnn_dropout_p=opt3["rnn_dropout_p"], bidirectional=bool(opt3["bidirectional"]))
model5 = S2VTAttModel(encoder, decoder).cuda()
model5.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_msvd_res152_i3d/model_320.pth').items()})

encoder = EncoderRNN(opt3["dim_vid"], opt3["dim_hidden"], bidirectional=bool(opt3["bidirectional"]),
                             input_dropout_p=opt3["input_dropout_p"], rnn_dropout_p=opt3["rnn_dropout_p"])
decoder = DecoderRNN(opt3["vocab_size"], opt3["max_len"], opt3["dim_hidden"], opt3["dim_word"],
                             input_dropout_p=opt3["input_dropout_p"],
                             rnn_dropout_p=opt3["rnn_dropout_p"], bidirectional=bool(opt3["bidirectional"]))
model6 = S2VTAttModel(encoder, decoder).cuda()
model6.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save_model/vatex_msrvtt_msvd_res152_i3d/model_200.pth').items()})

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

results = {}
results['result'] = []
results['version'] = "VERSION 1.2"
results['external_data'] = {}
results['external_data']['used'] = 'true'
results['external_data']['details'] = 'We used VATEX and MSRVTT and MSVD datasets to train the model'

test_list = os.listdir('./data/test/')

for i in test_list:
    v = {}
    fc_feat = []
    fc_feat.append(np.load('./data/test_res152_20_feats/' + i[:-4] +'.npy'))
    fc_feat = np.concatenate(fc_feat, axis=1)
    c3d_feat = np.load('./data/test_i3d_feats/' + i[:-4] +'.npy')
    c3d_feat = c3d_feat.reshape(-1, 1024)
    fc_feat = np.concatenate((fc_feat, c3d_feat), axis=1)
    feat = torch.from_numpy(fc_feat).type(torch.FloatTensor)
    feat = feat.cuda()
    feat = feat.reshape((1,-1,3072))
    seq_probs1, seq_preds1 = model1(feat, mode='inference', opt=opt1)
    sents1 = utils.decode_sequence(dataset1.get_vocab(), seq_preds1)
    seq_probs2, seq_preds2 = model2(feat, mode='inference', opt=opt1)
    sents2 = utils.decode_sequence(dataset1.get_vocab(), seq_preds2)
    if seq_probs1.sum() > seq_probs2.sum():
        sent = sents1
    else:
        sent = sents2
    v['video_id'] = i[:i.find('.')]
    v['caption'] = sent[0] + '.'
    results['result'].append(v)

for num,i in enumerate(test_list):
    fc_feat = []
    fc_feat.append(np.load('./data/test_res152_40_feats/' + i[:-4] +'.npy'))
    fc_feat = np.concatenate(fc_feat, axis=1)
    feat = torch.from_numpy(fc_feat).type(torch.FloatTensor)
    feat = feat.cuda()
    feat = feat.reshape((1,-1,2048))
    seq_probs3, seq_preds3 = model3(feat, mode='inference', opt=opt2)
    sents3 = utils.decode_sequence(dataset2.get_vocab(), seq_preds3)
    seq_probs4, seq_preds4 = model4(feat, mode='inference', opt=opt2)
    sents4 = utils.decode_sequence(dataset2.get_vocab(), seq_preds4)
    results['result'][num]['caption'] += sents3[0] + '.' + sents4[0] + '.'

for num,i in enumerate(test_list):
    fc_feat = []
    fc_feat.append(np.load('./data/test_res152_20_feats/' + i[:-4] +'.npy'))
    fc_feat = np.concatenate(fc_feat, axis=1)
    c3d_feat = np.load('./data/test_i3d_feats/' + i[:-4] +'.npy')
    c3d_feat = c3d_feat.reshape(-1, 1024)
    fc_feat = np.concatenate((fc_feat, c3d_feat), axis=1)
    feat = torch.from_numpy(fc_feat).type(torch.FloatTensor)
    feat = feat.cuda()
    feat = feat.reshape((1,-1,3072))
    seq_probs5, seq_preds5 = model5(feat, mode='inference', opt=opt3)
    sents5 = utils.decode_sequence(dataset3.get_vocab(), seq_preds5)
    seq_probs6, seq_preds6 = model6(feat, mode='inference', opt=opt3)
    sents6 = utils.decode_sequence(dataset3.get_vocab(), seq_preds6)
    results['result'][num]['caption'] += sents5[0] + '.' + sents6[0] + '.'

json.dump(results, open('submit/result.json', 'w')) 