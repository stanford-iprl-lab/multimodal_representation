from __future__ import print_function
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from models.sensor_fusion import SensorFusionSelfSupervised
from utils import (
    kl_normal,
    realEPE,
    compute_accuracy,
    flow2rgb,
    set_seeds,
    augment_val,
)

from dataloaders import ProcessForce, ToTensor
from dataloaders import MultimodalManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class selfsupervised:
    def __init__(self, configs, logger):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = configs["cuda"] and torch.cuda.is_available()

        self.configs = configs
        self.logger = logger
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            logger.print("Let's use", torch.cuda.device_count(), "GPUs!")

        set_seeds(configs["seed"], use_cuda)

        # model
        self.model = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            action_dim=configs["action_dim"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )

        self.deterministic = configs["deterministic"]
        self.encoder = configs["encoder"]

        # losses
        self.loss_ee_pos = nn.MSELoss()
        self.loss_contact_next = nn.BCEWithLogitsLoss()
        self.loss_optical_flow_mask = nn.BCEWithLogitsLoss()
        self.loss_reward_prediction = nn.MSELoss()
        self.loss_is_paired = nn.BCEWithLogitsLoss()
        self.loss_dynamics = nn.MSELoss()

        # validation set variables
        self.val_contact_accuracy = 0.0
        self.val_paired_accuracy = 0.0

        # test set variables
        self.test_flow_loss = 0.0
        self.test_paired_accuracy = 0.0
        self.test_contact_accuracy = 0.0

        # Weights for loss
        self.alpha_optical_flow = 10.0 * configs["opticalflow"]
        self.alpha_optical_flow_mask = 1.0
        self.alpha_kl = 0.05
        self.alpha_contact = 1.0 * configs["contact"]
        self.alpha_pair = 0.5 * configs["pairing"]
        self.alpha_ee_fut = 1.0 * configs["eedelta"]

        # Weights for input
        self.alpha_vision = configs["vision"]
        self.alpha_depth = configs["depth"]
        self.alpha_proprio = configs["proprio"]
        self.alpha_force = configs["force"]

        # Global Counts For Logging
        self.global_cnt = {"train": 0, "val": 0}

        # ------------------------
        # Handles Initialization
        # ------------------------
        if configs["load"]:
            self.load_model(configs["load"])

        self._init_dataloaders()

    def train(self):

        for i_epoch in tqdm(range(self.configs["max_epoch"])):
            # ---------------------------
            # Train Step
            # ---------------------------
            self.logger.print("Training epoch #{}...".format(i_epoch))
            self.model.train()

            for i_iter, sample_batched in tqdm(enumerate(self.dataloaders["val"])):

                t_st = time.time()
                self.optimizer.zero_grad()

                loss, mm_feat, results, image_packet = self.loss_calc(sample_batched)

                loss.backward()
                self.optimizer.step()

                self.record_results(loss, results, self.global_cnt["train"], t_st)

                if self.global_cnt["train"] % self.configs["img_record_n"] == 0:
                    self.logger.print(
                        "processed {} mini-batches...".format(self.global_cnt["train"])
                    )
                    self._record_image(image_packet, self.global_cnt["train"])

                self.global_cnt["train"] += 1

            if self.configs["val_ratio"] != 0:
                self.validate(i_epoch)

            # ---------------------------
            # Save weights
            # ---------------------------
            ckpt_path = os.path.join(
                self.logger.log_folder, "models", "weights_itr_{}.ckpt".format(i_epoch)
            )
            self.logger.print("checkpoint path: ", ckpt_path)
            self.logger.print("Saving checkpoint after epoch #{}".format(i_epoch))

            torch.save(self.model.state_dict(), ckpt_path)
            self.logger.end_itr(ckpt_path)

    def validate(self, i_epoch):
        self.logger.print(
            "calculating validation results after #{} epochs".format(i_epoch)
        )

        self.val_contact_accuracy = 0.0
        self.val_paired_accuracy = 0.0

        for i_iter, sample_batched in enumerate(self.dataloaders["val"]):
            self.model.eval()

            loss_val, mm_feat_val, results_val, image_packet_val = self.loss_calc(
                sample_batched
            )

            flow_loss, contact_loss, is_paired_loss, contact_accuracy, is_paired_accuracy, ee_delta_loss, kl = (
                results_val
            )

            self.val_contact_accuracy += contact_accuracy.item() / self.val_len_data
            self.val_paired_accuracy += is_paired_accuracy.item() / self.val_len_data

            if i_iter == 0:
                self._record_image(
                    image_packet_val, self.global_cnt["val"], string="val/"
                )

                self.logger.tb.add_scalar("val/loss/optical_flow", flow_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/contact", contact_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/is_paired", is_paired_loss.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/kl", kl.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/total_loss", loss_val.item(), self.global_cnt["val"])
                self.logger.tb.add_scalar("val/loss/ee_delta", ee_delta_loss, self.global_cnt["val"])
                self.global_cnt["val"] += 1

        # ---------------------------
        # Record Epoch Level Variables
        # ---------------------------
        self.logger.tb.add_scalar(
            "val/accuracy/contact", self.val_contact_accuracy, self.global_cnt["val"]
        )
        self.logger.tb.add_scalar(
            "val/accuracy/is_paired", self.val_paired_accuracy, self.global_cnt["val"]
        )

    def load_model(self, path):
        self.logger.print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def get_encoded_vals(self, sample_batched, paired=True):
        if not self.encoder:
            raise ValueError("Needs to be in encoder mode")

        if paired is True:
            # input data
            image = self.alpha_vision * sample_batched["image"].to(self.device)
            force = self.alpha_force * sample_batched["force"].to(self.device)
            proprio = self.alpha_proprio * sample_batched["proprio"].to(self.device)
            depth = self.alpha_depth * sample_batched["depth"].to(
                self.device
            ).transpose(1, 3).transpose(2, 3)

            action = sample_batched["action"].to(self.device)

            return self.model(image, force, proprio, depth, action)

        else:
            action = sample_batched["action"].to(self.device)
            unpaired_image = sample_batched["unpaired_image"].to(self.device)
            unpaired_force = sample_batched["unpaired_force"].to(self.device)
            unpaired_proprio = sample_batched["unpaired_proprio"].to(self.device)
            unpaired_depth = (
                sample_batched["unpaired_depth"]
                .to(self.device)
                .transpose(1, 3)
                .transpose(2, 3)
            )

            return self.model(
                unpaired_image, unpaired_force, unpaired_proprio, unpaired_depth, action
            )

    def loss_calc(self, sampled_batched):

        # input data
        image = self.alpha_vision * sampled_batched["image"].to(self.device)
        force = self.alpha_force * sampled_batched["force"].to(self.device)
        proprio = self.alpha_proprio * sampled_batched["proprio"].to(self.device)
        depth = self.alpha_depth * sampled_batched["depth"].to(self.device).transpose(
            1, 3
        ).transpose(2, 3)

        action = sampled_batched["action"].to(self.device)

        contact_label = sampled_batched["contact_next"].to(self.device)
        optical_flow_label = sampled_batched["flow"].to(self.device)
        optical_flow_mask_label = sampled_batched["flow_mask"].to(self.device)

        # unpaired data for sampled point
        unpaired_image = self.alpha_vision * sampled_batched["unpaired_image"].to(
            self.device
        )
        unpaired_force = self.alpha_force * sampled_batched["unpaired_force"].to(
            self.device
        )
        unpaired_proprio = self.alpha_proprio * sampled_batched["unpaired_proprio"].to(
            self.device
        )
        unpaired_depth = self.alpha_depth * sampled_batched["unpaired_depth"].to(
            self.device
        ).transpose(1, 3).transpose(2, 3)

        # labels to predict
        gt_ee_pos_delta = sampled_batched["ee_yaw_next"].to(self.device)

        if self.deterministic:
            paired_out, contact_out, flow2, optical_flow2_mask, ee_delta_out, mm_feat = self.model(
                image, force, proprio, depth, action
            )
            kl = torch.tensor([0]).to(self.device).type(torch.cuda.FloatTensor)
        else:
            paired_out, contact_out, flow2, optical_flow2_mask, ee_delta_out, mm_feat, mu_z, var_z, mu_prior, var_prior = self.model(
                image, force, proprio, depth, action
            )
            kl = self.alpha_kl * torch.mean(
                kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0))
            )

        flow_loss = self.alpha_optical_flow * realEPE(
            flow2, optical_flow_label, self.device
        )

        # Scene flow losses

        b, _, h, w = optical_flow_label.size()

        optical_flow_mask = nn.functional.upsample(
            optical_flow2_mask, size=(h, w), mode="bilinear"
        )

        flow_mask_loss = self.alpha_optical_flow_mask * self.loss_optical_flow_mask(
            optical_flow_mask, optical_flow_mask_label
        )
        

        contact_loss = self.alpha_contact * self.loss_contact_next(
            contact_out, contact_label
        )

        ee_delta_loss = self.alpha_ee_fut * self.loss_ee_pos(
            ee_delta_out, gt_ee_pos_delta
        )

        paired_loss = self.alpha_pair * self.loss_is_paired(
            paired_out, torch.ones(paired_out.size(0), 1).to(self.device)
        )

        unpaired_total_losses = self.model(
            unpaired_image, unpaired_force, unpaired_proprio, unpaired_depth, action
        )
        unpaired_out = unpaired_total_losses[0]
        unpaired_loss = self.alpha_pair * self.loss_is_paired(
            unpaired_out, torch.zeros(unpaired_out.size(0), 1).to(self.device)
        )

        loss = (
            contact_loss
            + paired_loss
            + unpaired_loss
            + ee_delta_loss
            + kl
            + flow_loss
            + flow_mask_loss
        )

        contact_pred = nn.Sigmoid()(contact_out).detach()
        contact_accuracy = compute_accuracy(contact_pred, contact_label.detach())

        paired_pred = nn.Sigmoid()(paired_out).detach()
        paired_accuracy = compute_accuracy(
            paired_pred, torch.ones(paired_pred.size()[0], 1, device=self.device)
        )

        unpaired_pred = nn.Sigmoid()(unpaired_out).detach()
        unpaired_accuracy = compute_accuracy(
            unpaired_pred, torch.zeros(unpaired_pred.size()[0], 1, device=self.device)
        )

        is_paired_accuracy = (paired_accuracy + unpaired_accuracy) / 2.0

        # logging
        is_paired_loss = paired_loss + unpaired_loss

        return (
            loss,
            mm_feat,
            (
                flow_loss,
                contact_loss,
                is_paired_loss,
                contact_accuracy,
                is_paired_accuracy,
                ee_delta_loss,
                kl,
            ),
            (flow2, optical_flow_label, image),
        )

    def record_results(self, total_loss, results, global_cnt, t_st):

        flow_loss, contact_loss, is_paired_loss, contact_accuracy, is_paired_accuracy, ee_delta_loss, kl = (
            results
        )

        self.logger.tb.add_scalar("loss/optical_flow", flow_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/contact", contact_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/is_paired", is_paired_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/kl", kl.item(), global_cnt)
        self.logger.tb.add_scalar("loss/total_loss", total_loss.item(), global_cnt)
        self.logger.tb.add_scalar("loss/ee_delta", ee_delta_loss, global_cnt)

        self.logger.tb.add_scalar(
            "accuracy/contact", contact_accuracy.item(), global_cnt
        )
        self.logger.tb.add_scalar(
            "accuracy/is_paired", is_paired_accuracy.item(), global_cnt
        )

        self.logger.tb.add_scalar("stats/iter_time", time.time() - t_st, global_cnt)

    def _init_dataloaders(self):

        filename_list = []
        for file in os.listdir(self.configs["dataset"]):
            if file.endswith(".h5"):
                filename_list.append(self.configs["dataset"] + file)

        self.logger.print(
            "Number of files in multifile dataset = {}".format(len(filename_list))
        )

        # TODO Figure out how to clean up this section

        val_filename_list = []

        val_index = np.random.randint(
            0, len(filename_list), int(len(filename_list) * self.configs["val_ratio"])
        )

        for index in val_index:
            val_filename_list.append(filename_list[index])

        while val_index.size > 0:
            filename_list.pop(val_index[0])
            val_index = np.where(val_index > val_index[0], val_index - 1, val_index)
            val_index = val_index[1:]

        self.logger.print("Initial finished")

        val_filename_list1, filename_list1 = augment_val(
            val_filename_list, filename_list
        )

        self.logger.print("Listing finished")

        # TODO: Can be cleaned up

        self.dataloaders = {}
        self.samplers = {}
        self.datasets = {}

        self.samplers["val"] = SubsetRandomSampler(
            range(len(val_filename_list1) * (self.configs["ep_length"] - 1))
        )
        self.samplers["train"] = SubsetRandomSampler(
            range(len(filename_list1) * (self.configs["ep_length"] - 1))
        )

        self.logger.print("Sampler finished")

        self.datasets["train"] = MultimodalManipulationDataset(
            filename_list1,
            transform=transforms.Compose(
                [
                    ProcessForce(32, "force", tanh=True),
                    ProcessForce(32, "force_fut", tanh=True),
                    ProcessForce(32, "unpaired_force", tanh=True),
                    ProcessForce(32, "force_cp", tanh=True),
                    ProcessForce(32, "force_fp", tanh=True),
                    ToTensor(device=self.device),
                ]
            ),
            episode_length=self.configs["ep_length"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],

        )

        self.datasets["val"] = MultimodalManipulationDataset(
            val_filename_list1,
            transform=transforms.Compose(
                [
                    ProcessForce(32, "force", tanh=True),
                    ProcessForce(32, "force_fut", tanh=True),
                    ProcessForce(32, "unpaired_force", tanh=True),
                    ProcessForce(32, "force_cp", tanh=True),
                    ProcessForce(32, "force_fp", tanh=True),
                    ToTensor(device=self.device),
                ]
            ),
            episode_length=self.configs["ep_length"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],

        )

        self.logger.print("Dataset finished")

        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            sampler=self.samplers["val"],
            pin_memory=True,
            drop_last=True,
        )
        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            sampler=self.samplers["train"],
            pin_memory=True,
            drop_last=True,
        )

        self.len_data = len(self.dataloaders["train"])
        self.val_len_data = len(self.dataloaders["val"])

        self.logger.print("Finished setting up date")

    def _record_image(self, image_packet, global_cnt, string=None):

        if string is None:
            string = ""

        flow2, flow_label, image = image_packet
        image_index = 0

        b, c, h, w = flow_label.size()

        upsampled_flow = nn.functional.upsample(flow2, size=(h, w), mode="bilinear")
        upsampled_flow = upsampled_flow.cpu().detach().numpy()
        orig_image = image[image_index].cpu().numpy()  # .transpose(1,2,0)

        orig_flow = flow2rgb(
            flow_label[image_index].cpu().detach().numpy(), max_value=None
        )
        pred_flow = flow2rgb(upsampled_flow[image_index], max_value=None)

        concat_image = np.concatenate([orig_image, orig_flow, pred_flow], 1)

        concat_image = concat_image * 255
        concat_image = concat_image.astype(np.uint8)
        concat_image = concat_image.transpose(2, 0, 1)

        self.logger.tb.add_image(string + "predicted_flow", concat_image, global_cnt)
