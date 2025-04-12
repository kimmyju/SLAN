import torch
import numpy
import logging
import os
import torch.nn.functional as F

from utils.dice_score import multiclass_dice_coeff
from utils.dice_score import dice_loss

class Trainer(object) : 
    
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            val_step,
            optimizer,
            criterion,
            device,
            scheduler = None,
            grad_scaler = None,
            checkpoint_step = 10,
            checkpoint_path = "./",
    ) :
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_step = val_step
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.checkpoint_step = checkpoint_step
        self.checkpoint_path = checkpoint_path
        self.max_epoch = 0 
        self.epoch = 1
        self.epoch_loss = 0

        self.obj_score = 0
        self.sha_score = 0
        self.obj_best_score = 0
        self.sha_best_score = 0

    def _save_checkpoint(self) :
        if (not os.path.exists(self.checkpoint_path)) :
            os.makedirs(self.checkpoint_path)
        file_path = os.path.join(self.checkpoint_path, "checkpoint_{}.pth".format(self.epoch))
        torch.save(self.model.state_dict(), file_path)

    def _load_checkpoint(self) :
        pass

    def _t(self) :
        self.model.train()
        for batch in self.train_loader :
            imgs = batch["image"].to(device = self.device, dtype = torch.float32, memory_format = torch.channels_last)
            obj_masks = batch["obj_mask"].to(device = self.device, dtype = torch.long)
            sha_masks = batch["sha_mask"].to(device = self.device, dtype = torch.long)
            light = int(batch["light"])
            with torch.autocast(self.device.type if self.device.type != "mps" else "cpu", enabled = True) :
                obj_preds, sha_preds = self.model([imgs, light])
                obj_loss = self.criterion(obj_preds, obj_masks)
                sha_loss = self.criterion(sha_preds, sha_masks)

                obj_loss += dice_loss(
                    F.softmax(obj_preds, dim = 1).float(),
                    F.one_hot(obj_masks, self.model.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass = True
                )
                sha_loss += dice_loss(
                    F.softmax(sha_preds, dim = 1).float(),
                    F.one_hot(sha_masks, self.model.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass = True
                )

                loss = (0.5 * obj_loss) + (0.5 * sha_loss)
                self.optimizer.zero_grad(set_to_none = True)
                self.grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.epoch_loss += loss.item()
        print("[epoch {:04d}] loss : {:.5f}".format(self.epoch, self.epoch_loss / len(self.train_loader)))
        self.epoch_loss = 0 

    def _v(self) :
        self.model.eval()
        num_val_batches = len(self.val_loader)
        obj_dice_score = 0
        sha_dice_score = 0
        with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled = True) :
            for batch in self.val_loader :
                imgs = batch["image"].to(device = self.device, dtype = torch.float32, memory_format = torch.channels_last)
                obj_masks = batch["obj_mask"].to(device = self.device, dtype = torch.long)
                sha_masks = batch["sha_mask"].to(device = self.device, dtype = torch.long)
                light = int(batch["light"])
                obj_preds, sha_preds = self.model([imgs, light])

                obj_masks = F.one_hot(obj_masks, self.model.num_classes).permute(0, 3, 1, 2).float()
                obj_preds = F.one_hot(obj_preds.argmax(dim = 1), self.model.num_classes).permute(0, 3, 1, 2).float()

                sha_masks = F.one_hot(sha_masks, self.model.num_classes).permute(0, 3, 1, 2).float()
                sha_preds = F.one_hot(sha_preds.argmax(dim = 1), self.model.num_classes).permute(0, 3, 1, 2).float()

                obj_dice_score += multiclass_dice_coeff(obj_preds[:, 1:], obj_masks[:, 1:], reduce_batch_first = False)
                sha_dice_score += multiclass_dice_coeff(sha_preds[:, 1:], sha_masks[:, 1:], reduce_batch_first = False)

        return obj_dice_score / max(num_val_batches, 1), sha_dice_score / max(num_val_batches, 1)
 
    def _excute_one_epoch(self) :
        self._t()
        if (self.epoch % self.val_step == 0) :
            with torch.no_grad() :
                self.obj_score, self.sha_score = self._v()
                score = self.obj_score + self.sha_score
                self.scheduler.step(score)
                print("object score : {:.5f} shadow score : {:.5f}".format(self.obj_score, self.sha_score))
                if (self.obj_best_score < self.obj_score.data) :
                    self.obj_best_score = self.obj_score.data
                if (self.sha_best_score < self.sha_score.data) :
                    self.sha_best_score = self.sha_score.data
                print("best object score : {:.5f} best shadow score : {:.5f}".format(self.obj_best_score, self.sha_best_score))

    def run(self, max_epoch) :
        self.max_epoch = max_epoch
        for epoch in range(self.epoch, self.max_epoch + 1) :
            self.epoch = epoch 
            self._excute_one_epoch()
            if (self.epoch % self.checkpoint_step == 0) :
                logging.info(f"saving checkpoint")
                self._save_checkpoint()
        logging.info(f"finished training")