from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter

class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError



class Trainer(BaseTrainer):
    """

        training baseline include softmax and triplet loss

    """
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)

            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)

        return loss, prec



# class Trainer_softmax_triplet(BaseTrainer):
#     """
#         training model which combine softmax and triplet
#     """
#     def _parse_data(self, inputs):
#         imgs, _, pids, _ = inputs
#         inputs = [Variable(imgs)]
#         targets = Variable(pids.cuda())
#
#         return inputs, targets
#
#     def _forward(self,inputs,targets):
#         outputs_t,outputs_s=self.model(*inputs)
#
#         total_loss=torch.FloatTensor([0.0]).squeeze(0).cuda()
#         total_prec=torch.FloatTensor([0.0]).squeeze(0).cuda()
#
#         assert len(self.criterion) == 2, 'criterions size should be 2 !'
#
#         if isinstance(self.criterion[0], torch.nn.CrossEntropyLoss):
#             loss = self.criterion[0](outputs_t, targets)
#             prec, = accuracy(outputs_t.data, targets.data)
#             prec = prec[0]
#         elif isinstance(self.criterion[0], OIMLoss):
#             loss, outputs = self.criterion[0](outputs_t, targets)
#             prec, = accuracy(outputs.data, targets.data)
#             prec = prec[0]
#         elif isinstance(self.criterion[0], TripletLoss):
#             loss, prec = self.criterion[0](outputs_t, targets)
#         else:
#             raise ValueError("Unsupported loss:", self.criterion)
#
#         total_loss += loss
#         total_prec += prec
#
#         if isinstance(self.criterion[1], torch.nn.CrossEntropyLoss):
#             loss = self.criterion[1](outputs_s, targets)
#             prec, = accuracy(outputs_s.data, targets.data)
#             prec = prec[0]
#         elif isinstance(self.criterion[1], OIMLoss):
#             loss, outputs = self.criterion[1](outputs_s, targets)
#             prec, = accuracy(outputs.data, targets.data)
#             prec = prec[0]
#         elif isinstance(self.criterion[1], TripletLoss):
#             loss, prec = self.criterion[1](outputs_s, targets)
#         else:
#             raise ValueError("Unsupported loss:", self.criterion)
#
#         total_loss += loss
#         total_prec += prec
#
#         return total_loss,total_prec/2.0



class Trainer_softmax_triplet(object):
    def __init__(self, model, criterions, print_freq=1):
        super(Trainer_softmax_triplet, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
            losses.update(loss.data[0], targets.size(0))

            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)

        #new added by wc
        # x1 triplet loss
        loss_tri, prec_tri = self.criterions[0](outputs[0], targets)
        # x2 global feature cross entropy loss
        loss_global = self.criterions[1](outputs[1], targets)

        if isinstance(loss_global,tuple):
            loss_global, prec_global = loss_global

        prec_global = accuracy(outputs[1].data, targets.data)
        prec_global = prec_global[0][0]

        return loss_tri+loss_global, prec_global



from tensorboardX import SummaryWriter
writer=SummaryWriter()

class Trainer_pcb(BaseTrainer):
    """
        training  pcb model

    """
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        feats_list,logits_list = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = torch.sum(torch.stack([self.criterion(logits, targets) for logits in logits_list]))
            prec = accuracy(logits_list[0].detach(), targets.detach())
            prec=prec[0][0]

        # elif isinstance(self.criterion, OIMLoss):
        #     loss, outputs = self.criterion(outputs, targets)
        #     prec = accuracy(outputs.data, targets.data)
        #     prec = prec[0]
        # elif isinstance(self.criterion, TripletLoss):
        #     loss, prec = self.criterion(outputs, targets)
        else:
            total_feat = torch.stack(feats_list)
            total_feat = total_feat.permute(1, 0, 2).contiguous()
            total_feat = total_feat.view(total_feat.size(0), -1)
            print(total_feat.size())
            loss, prec = self.criterion(total_feat, targets.detach())


        return loss, prec


    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
            writer.add_scalar('Train/Loss',loss.detach(),epoch)
            writer.add_scalar('Train/Prec1',prec1.detach(),epoch)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

