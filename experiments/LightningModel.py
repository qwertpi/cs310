from __future__ import annotations

from pytorch_lightning import LightningModule
import torch

from ModelEvaluator import ER_POS_PREVALANCE, PR_POS_PREVALANCE


class LightningModel(LightningModule):
    def __init__(
        self,
        torch_model: torch.nn.Module,
        weight_decay: float,
        remove_label_correlations: bool,
        discard_conflicting_labels: bool,
    ):
        super().__init__()
        self.model = torch_model
        self.weight_decay = weight_decay
        self.remove_label_correlations = remove_label_correlations
        self.discard_conflicting_labels = discard_conflicting_labels

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_losses(self, batch):
        er_labels = batch.y[:, 0]
        pr_labels = batch.y[:, 1]
        out = self.model(batch.x, batch.edge_index, batch.batch)
        er_pred = out[:, 0]
        pr_pred = out[:, 1]

        # We drop ER-PR+ cases to avoid confusing the model
        er_labels_given_pr_pos = er_labels[(pr_labels == 1) & (er_labels == 1)]
        pr_labels_given_er_neg = pr_labels[(er_labels == 0) & (pr_labels == 0)]
        if not self.discard_conflicting_labels:
            er_labels_given_pr_neg = er_labels[pr_labels == 0]
            pr_labels_given_er_pos = pr_labels[er_labels == 1]
        else:
            er_labels_given_pr_neg = er_labels[(er_labels == 0) & (pr_labels == 0)]
            pr_labels_given_er_pos = pr_labels[(pr_labels == 1) & (er_labels == 1)]

        er_pred_given_pr_pos = er_pred[(pr_labels == 1) & (er_labels == 1)]
        pr_pred_given_er_neg = pr_pred[(er_labels == 0) & (pr_labels == 0)]
        if not self.discard_conflicting_labels:
            er_pred_given_pr_neg = er_pred[pr_labels == 0]
            pr_pred_given_er_pos = pr_pred[er_labels == 1]
        else:
            er_pred_given_pr_neg = er_pred[(er_labels == 0) & (pr_labels == 0)]
            pr_pred_given_er_pos = pr_pred[(pr_labels == 1) & (er_labels == 1)]

        # Oversampling in the BCE losses so positive and negative examples count equally
        er_pos_weight = torch.tensor(ER_POS_PREVALANCE)
        pr_pos_weight = torch.tensor(PR_POS_PREVALANCE)

        batch_er_loss_given_pr_pos = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                er_pred_given_pr_pos,
                er_labels_given_pr_pos,
                pos_weight=er_pos_weight,
            )
        ).nan_to_num(0)
        batch_er_loss_given_pr_neg = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                er_pred_given_pr_neg,
                er_labels_given_pr_neg,
                pos_weight=er_pos_weight,
            )
        ).nan_to_num(0)

        batch_pr_loss_given_er_pos = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pr_pred_given_er_pos,
                pr_labels_given_er_pos,
                pos_weight=pr_pos_weight,
            )
        ).nan_to_num(0)
        batch_pr_loss_given_er_neg = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pr_pred_given_er_neg,
                pr_labels_given_er_neg,
                pos_weight=pr_pos_weight,
            )
        ).nan_to_num(0)

        if self.remove_label_correlations:
            # If we had done one big call to BCE, these groups of terms would
            # have been skewed by the uneven distribution of
            # the other label, but here we weight them equally
            er_pr_pos_scale = er_pr_neg_scale = pr_er_pos_scale = pr_er_neg_scale = 0.5
        else:
            # Exactly sample in proportion to the other label, as if we had done one big call to BCE
            er_pr_pos_scale = len(er_labels_given_pr_pos) / (
                len(er_labels_given_pr_pos) + len(er_labels_given_pr_neg)
            )
            er_pr_neg_scale = len(er_labels_given_pr_neg) / (
                len(er_labels_given_pr_pos) + len(er_labels_given_pr_neg)
            )
            pr_er_pos_scale = len(pr_labels_given_er_pos) / (
                len(pr_labels_given_er_pos) + len(pr_labels_given_er_neg)
            )
            pr_er_neg_scale = len(pr_labels_given_er_neg) / (
                len(pr_labels_given_er_pos) + len(pr_labels_given_er_neg)
            )

        batch_er_loss = (
            er_pr_pos_scale * batch_er_loss_given_pr_pos
            + er_pr_neg_scale * batch_er_loss_given_pr_neg
        )
        # Weight ER+, and ER- cases equally whereas if we had done
        # one big call to BCE they would have been skewed by the uneven
        # distribution of the ER label
        batch_pr_loss = (
            pr_er_pos_scale * batch_pr_loss_given_er_pos
            + pr_er_neg_scale * batch_pr_loss_given_er_neg
        )

        loss = batch_er_loss + batch_pr_loss
        return batch_er_loss, batch_pr_loss, loss

    def training_step(self, batch):
        batch_er_loss, batch_pr_loss, loss = self.compute_losses(batch)
        self.log(
            "train_er_loss",
            batch_er_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_pr_loss",
            batch_pr_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch):
        batch_er_loss, batch_pr_loss, loss = self.compute_losses(batch)
        self.log(
            "val_er_loss",
            batch_er_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_pr_loss",
            batch_pr_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log("val_loss", loss, batch_size=len(batch), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        # PyTorch's default weight decay decays parameters that should not be regularised
        # This code based on: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/14
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for module in self.modules():
            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.PReLU,
                ),
            ):
                no_decay_params.extend(module.parameters(recurse=False))
            else:
                for name, param in module.named_parameters(recurse=False):
                    if "bias" in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

        if sum((p.numel() for p in no_decay_params)) + sum(
            (p.numel() for p in decay_params)
        ) != sum((p.numel() for p in self.parameters(recurse=True))):
            raise RuntimeError("Lost or gained paramaters")

        return torch.optim.AdamW(
            (
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0},
            )
        )
