import os
import time
import math
import pprint
import shutil
import random
import inspect
import pickle
from collections import OrderedDict, defaultdict

import yaml
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch_edit_distance import levenshtein_distance
from src.globals import PHRASE_PAD_VALUE, PHRASE_EOS_IDX

from src.metrics.TopKAccuracy import TopKAccuracy
from utils import import_class, count_params
from IPython import embed; from sys import exit

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer():
    
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.y:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')

        self.lr = self.arg.base_lr
        self.best_loss = float('inf')
        self.global_step = 0
        self.best_loss_epoch = 0
        self.load_data()
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )
                self.translate_fn = self.model.module.greedy_translate_sequence
        self.topk_acc = TopKAccuracy(topk=(1,5))

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        Encoder = import_class(self.arg.encoder)
        Decoder = import_class(self.arg.decoder)
        # Copy model file and main
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(inspect.getfile(Encoder), self.arg.work_dir)
        shutil.copy2(inspect.getfile(Decoder), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        self.model = Model(
            Encoder(**self.arg.encoder_args), 
            Decoder(**self.arg.decoder_args),
            **self.arg.model_args
        ).cuda(output_device)
        # from torchinfo import summary
        # summary(self.model, input_size=[(16, 128, 164), (16,32)], dtypes=[torch.float32, torch.int32], verbose=2)
        self.translate_fn = self.model.greedy_translate_sequence
        self.loss = nn.CrossEntropyLoss(**self.arg.loss_args)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        # for p in self.model.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, min=None, max=5.))

        # check if model contains late dropout and set the steps per epoch prop
        if hasattr(self.model.decoder, 'late_dropout') and self.arg.phase == 'train':
            steps_per_epoch = len(self.data_loader['train'].dataset) // self.arg.batch_size
            self.model.decoder.late_dropout.num_steps_per_epoch = steps_per_epoch

        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        if hasattr(self.model, 'finetune_params'):
            self.param_groups['finetune_params'].extend(self.model.finetune_params)
            self.param_groups['base_params'].extend(self.model.base_params)
            self.optim_param_groups = {
                'finetune': {'params': self.param_groups['finetune_params'], 'lr': self.arg.finetune_lr},
                'base': {'params': self.param_groups['base_params']},
            }
        else:
            for name, params in self.model.named_parameters():
                self.param_groups['other'].append(params)

            self.optim_param_groups = {
                'other': {'params': self.param_groups['other']}
            }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        optimizer_cls = import_class(self.arg.optimizer)
        optimizer_args = self.arg.optimizer_args
        optimizer_args['params'] = params
        optimizer_args['lr'] = self.lr
        self.wd_ratio = optimizer_args.get('weight_decay', 1.)
        self.optimizer = optimizer_cls(**optimizer_args)

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr:.10f} weight_decay: {self.optimzier.weight_decay}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        if self.arg.lr_scheduler is None:
            lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR
            self.arg.lr_scheduler_args = dict(
                lr_lambda=lambda epoch: 1, # multiplicative factor given the epoch
            )
        else:
            lr_scheduler_cls = import_class(self.arg.lr_scheduler)
        self.lr_scheduler = lr_scheduler_cls(optimizer=self.optimizer, **self.arg.lr_scheduler_args)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn,
                pin_memory=True,
                persistent_workers=self.arg.num_worker > 0,
            )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn,
            pin_memory=True,
            persistent_workers=self.arg.num_worker > 0
        )
        
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def evaluation_metric(self, preds, labels, return_per_sample=False):
        """The evaluation metric for this contest is the normalized total 
        levenshtein distance. 
        
        Let the total number of characters in the labels be N and the total 
        levenshtein distance be D. The metric equals (N - D) / N.
        """
        # preds = (batch_size, phrase_len-1)
        # labels = (batch_size, phrase_len)
        # assume first position is <sos>
        preds = preds[:, 1:]
        labels = labels[:,1:]

        # get labels lengths (excluding <pad>)
        labels_len = ((labels != PHRASE_PAD_VALUE).sum(1)) - 1

        # get predictions lengths (up until first <eos> included)
        preds_len = (preds==PHRASE_EOS_IDX).float().argmax(dim=1)
        preds_len = torch.where(preds_len == 0, preds.shape[1], preds_len)
        
        preds = preds.to(torch.int32).cuda()
        labels = labels.to(torch.int32).cuda()
        preds_len = preds_len.to(torch.int32).cuda()
        labels_len = labels_len.to(torch.int32).cuda()
        placeholder = torch.tensor(-100, dtype=torch.int32).cuda()
        d_or = levenshtein_distance(
            hypotheses=preds, 
            references=labels, 
            hypothesis_lengths=preds_len, 
            references_lengths=labels_len, 
            blank=placeholder, 
            separator=placeholder
        )
        d = d_or[...,:3].sum(-1)
        n = d_or[...,3]
        norm_ed_dist = (n - d) / n
        mean_norm_ed_dist = norm_ed_dist.mean().item()
        mean_ed_dist = d.to(torch.float32).mean().item()
        std_norm_ed_dist = norm_ed_dist.std().item()
        std_ed_dist = d.to(torch.float32).std().item()
        conf_int_95_norm_ed_dist = 1.96 * std_norm_ed_dist / np.sqrt(len(preds))
        conf_int_95_ed_dist = 1.96 * std_ed_dist / np.sqrt(len(preds))

        toret = (mean_ed_dist, mean_norm_ed_dist, conf_int_95_ed_dist, conf_int_95_norm_ed_dist)
        if return_per_sample:
            toret += (d_or,)
        return toret
    
    def train(self, epoch, to_eval):
        self.model.train()

        self.topk_acc.reset()
        
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['weight_decay'] = current_lr * self.wd_ratio
        current_wd = self.optimizer.param_groups[0]['weight_decay']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.2E} WD: {current_wd:.2E}')

        epoch_preds = []
        epoch_labels = []
        process = tqdm(loader, dynamic_ncols=True)
        for (_, data, label) in process:
            self.global_step += 1
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]

                # forward
                output, _ = self.model(batch_data, batch_label[:, :-1])
                loss = self.loss(
                    output.permute(0,2,1),
                    batch_label[:, 1:],
                )
                # output_dim = output.shape[-1]
                # mask = torch.ones(output_dim).to(self.output_device)
                # mask[PHRASE_PAD_VALUE] = 0.
                # loss = loss * mask
                # loss = loss.mean()
                loss = loss / splits
                self.topk_acc.update(output, batch_label)

                loss.backward()

                if to_eval:
                    with torch.no_grad():
                        trans = self.translate_fn(batch_data)
                    # to compute training edit distance
                    epoch_preds.append(trans.detach().cpu().to(torch.uint8))
                    epoch_labels.append(batch_label.detach().cpu().to(torch.uint8))

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)

            #####################################
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        if to_eval:
            # compute edit distance
            epoch_preds = torch.vstack(epoch_preds)
            epoch_labels = torch.vstack(epoch_labels)
            edit_dist, norm_ed_dist, conf_int_95_ed_dist, conf_int_95_norm_ed_dist = self.evaluation_metric(epoch_preds, epoch_labels)
            self.train_writer.add_scalar('lev_dist', edit_dist, self.global_step)
            self.train_writer.add_scalar('norm_tot_lev_dist', norm_ed_dist, self.global_step)

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        topk_accs = self.topk_acc.get_result()
        self.print_log(f'\tMean training loss ({len(loader)} batches): {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log(f'\tMean training PPL: {math.exp(mean_loss):.4f} (BS: {math.exp(mean_loss) * num_splits:.4f}).')
        for k, v in topk_accs.items():
            self.print_log(f'\tTop-{k} acc: {v:.4f}')
        if to_eval:
            self.print_log(f'\tMean training ED-D: {edit_dist:.4f}±{conf_int_95_ed_dist:.4f} (BS: {edit_dist * num_splits:.4f}).')
            self.print_log(f'\tMean training N-ED-D: {norm_ed_dist:.4f}±{conf_int_95_norm_ed_dist:.4f} (BS: {norm_ed_dist * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        
        toret = (mean_loss,)
        if to_eval:
            toret += (edit_dist, norm_ed_dist,)
        
        return toret

    def eval(self, epoch, loader_name=['test'], wrong_file=None, result_file=None, display_predictions=False):
        self.model.eval()

        with torch.no_grad():
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                epoch_preds = []
                epoch_labels = []
                for _, data, label in process:
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output, _ = self.model(data, label[:, :-1])
                    output_dim = output.shape[-1]
                    # loss = self.loss(
                    #     output.contiguous().view(-1, output_dim), 
                    #     label[:, 1:].contiguous().view(-1)
                    # )
                    loss = self.loss(
                        output.permute(0,2,1),
                        label[:, 1:],
                    )
                    mask = torch.ones(output_dim).to(self.output_device)
                    mask[PHRASE_PAD_VALUE] = 0.
                    loss = loss * mask
                    loss = loss.mean()

                    trans = self.translate_fn(data)

                    loss_values.append(loss.cpu().item())
                    epoch_preds.append(trans.detach().cpu().to(torch.uint8))
                    epoch_labels.append(label.detach().cpu().to(torch.uint8))
                    step += 1

            loss = np.mean(loss_values)
            
            # compute edit distance
            epoch_preds = torch.vstack(epoch_preds)
            epoch_labels = torch.vstack(epoch_labels)
            
            edit_dist, norm_ed_dist, conf_int_95_ed_dist, conf_int_95_norm_ed_dist, per_sample = self.evaluation_metric(epoch_preds, epoch_labels, return_per_sample=True)

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_loss_epoch = epoch + 1
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('lev_dist', edit_dist, self.global_step)
                self.val_writer.add_scalar('norm_tot_lev_dist', norm_ed_dist, self.global_step)

            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {loss:.4f}.')
            self.print_log(f'\tMean PPL: {math.exp(loss):.4f}.')
            self.print_log(f'\tMean ED-D: {edit_dist:.4f}±{conf_int_95_ed_dist:.4f}.')
            self.print_log(f'\tMean N-ED-D: {norm_ed_dist:.4f}±{conf_int_95_norm_ed_dist:.4f}.')

        torch.cuda.empty_cache()
        
        return loss, edit_dist, norm_ed_dist, per_sample

    def display_translations(self, seq_idxs, seq, phrase, idx_to_char):
        trans = self.translate_fn(seq)
        _, _, _, _, ed_dist = self.evaluation_metric(trans, phrase, return_per_sample=True)
        seq_idxs = pd.DataFrame(seq_idxs.cpu().numpy())
        trans = pd.DataFrame(trans.cpu().numpy()).applymap(lambda x: idx_to_char[x]).apply(lambda x: ''.join(x), axis=1).replace('<eos>.*','', regex=True).str.replace('<sos>','')
        phrase = pd.DataFrame(phrase.cpu().numpy()).applymap(lambda x: idx_to_char[x]).replace('<pad>', '').apply(lambda x: ''.join(x), axis=1).replace('<eos>.*','', regex=True).str.replace('<sos>','')
        
        D = ed_dist[...,:3].cpu().sum(-1).numpy()
        N = ed_dist[...,3].cpu().numpy()
        phrase_len = phrase.str.len()
        trans_len = trans.str.len()
        edd = pd.DataFrame(D)
        nedd = pd.DataFrame((N-D)/N)
        toshow = pd.concat([seq_idxs, phrase, trans, edd, nedd, phrase_len, trans_len], axis=1)
        toshow.columns = ['seq_idx', 'phrase', 'pred', 'ed_dist', 'norm_ed_dist', 'phrase_len', 'pred_len']
        toshow = toshow.set_index('seq_idx')
        self.print_log(f'Predictions:\n{toshow.to_markdown(index=True)}')

    def start(self):
        if self.arg.phase == 'train':
            del self.arg.optimizer_args['params']
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                eval_model = self.arg.run_eval and (((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch))
                display_predictions = ((epoch + 1) % self.arg.display_predictions_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                
                self.train(epoch, eval_model)
                
                if eval_model:
                    self.eval(epoch, loader_name=['test'] )

                if save_model or (epoch + 1 == self.arg.num_epoch):
                    # save training checkpoint & weights
                    self.save_weights(epoch + 1)

                if save_model:
                    self.save_checkpoint(epoch + 1)

                if display_predictions:
                    seq_idxs, data, label = next(iter(self.data_loader['test']))
                    seq_idxs = seq_idxs[:100].cuda(self.output_device)
                    data = data[:100].cuda(self.output_device)
                    label = label[:100].cuda(self.output_device)

                    with torch.no_grad():
                        self.display_translations(seq_idxs=seq_idxs, seq=data, phrase=label, idx_to_char=self.data_loader['test'].dataset.idx_to_char)
    
                self.lr_scheduler.step()

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Base LR: {self.arg.base_lr:.10f}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                loader_name=['test'],
            )
            seq_idxs, data, label = next(iter(self.data_loader['test']))
            seq_idxs = seq_idxs[:100].cuda(self.output_device)
            data = data[:100].cuda(self.output_device)
            label = label[:100].cuda(self.output_device)
            self.display_translations(seq_idxs=seq_idxs, seq=data, phrase=label, idx_to_char=self.data_loader['test'].dataset.idx_to_char)