import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader
import argparse
import tqdm


def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    for epoch in range(opt["epochs"]):
        total_loss = 0
        epoch_loss = 0
        train_batch = 0
        
        #train_iterator = tqdm.tqdm(loader,ncols=40)
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            #labels = [i.cuda() for i in data['labels']]
            #masks = [i.cuda() for i in data['masks']]
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            epoch_loss = total_loss / (train_batch + 1)
            torch.cuda.synchronize()

            if not sc_flag:
                if (train_batch+1)%1 == 0:
                    print("(epoch %d), [iter %d], batch_loss = %.6f, epoch_loss = %.6f" %
                          (epoch, train_batch+1, loss.item(), total_loss / (train_batch + 1)))
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (train_batch, epoch, np.mean(reward[:, 0])))
            train_batch += 1
        
        lr_scheduler.step(epoch)

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            #if opt['local_rank'] == 0:
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, epoch_loss))

    
def main(opt):
    #torch.cuda.set_device(opt['local_rank'])
    #torch.distributed.init_process_group(
    #            backend='nccl',
    #            init_method='env://',
    #            )
    
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True, num_workers=8)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #dataloader = DataLoader(dataset, batch_size=opt["batch_size"], sampler=train_sampler, num_workers=8)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=bool(opt["bidirectional"]),
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=bool(opt["bidirectional"]))
        model = S2VTAttModel(encoder, decoder)
    model = model.cuda()    
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt['local_rank']], output_device=opt['local_rank'])
    #model.load_state_dict(torch.load('./save/all_data_res152_i3d/model_400.pth'))
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #model = torch.nn.parallel.DistributedDataParallel(model)
    
    #model.load_state_dict(torch.load('./save/res152_s2vtatt/model_460.pth'))
    #model = nn.DataParallel(model, device_ids=[0,1])
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    #optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    #os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
