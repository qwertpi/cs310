from __future__ import annotations

from pytorch_lightning import LightningModule
import torch

from ModelEvaluationUtils import (
    ER_POS_GIVEN_PR_NEG_PROB,
    ER_POS_GIVEN_PR_POS_PROB,
    ER_POS_PROB,
    PR_POS_GIVEN_ER_NEG_PROB,
    PR_POS_GIVEN_ER_POS_PROB,
    PR_POS_PROB,
)


class LightningModel(LightningModule):
    def __init__(
        self,
        torch_model: torch.nn.Module,
        weight_decay: float,
        remove_label_correlations: bool,
        discard_conflicting_labels: bool,
        spectral_decoupling: bool,
    ):
        super().__init__()
        self.model = torch_model
        self.weight_decay = weight_decay
        self.remove_label_correlations = remove_label_correlations
        self.discard_conflicting_labels = discard_conflicting_labels
        self.spectral_decoupling = spectral_decoupling

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_losses(self, batch):
        def _compute_subset_loss(pred, true, pos_prob):
            zero = torch.tensor([0], device="cuda")
            if pred.numel() == 0:
                return zero
            error = torch.nn.functional.binary_cross_entropy_with_logits(
                pred, true, pos_weight=torch.tensor((1 - pos_prob) / pos_prob)
            )
            if self.spectral_decoupling:
                confidences = torch.nn.functional.sigmoid(pred) - 0.5
                penalty = (confidences**2).mean()
            else:
                penalty = zero
            return error + 1e-3 * penalty

        er_labels = batch.y[:, 0]
        pr_labels = batch.y[:, 1]
        out = self.model(batch.x, batch.edge_index, batch.batch)
        if self.discard_conflicting_labels:
            mask = ((er_labels == 0) & (pr_labels == 0)) | (
                (er_labels == 1) & (pr_labels == 1)
            )
        # In any case we drop ER-PR+ cases to avoid confusing the model as these may not really exist
        else:
            mask = [(er_labels == 1) | (pr_labels == 0)]
        er_labels = er_labels[mask]
        pr_labels = pr_labels[mask]
        er_pred = out[:, 0][mask]
        pr_pred = out[:, 1][mask]

        if self.remove_label_correlations:
            er_labels_given_pr_pos = er_labels[(pr_labels == 1)]
            er_labels_given_pr_neg = er_labels[(pr_labels == 0)]
            pr_labels_given_er_pos = pr_labels[(er_labels == 1)]
            pr_labels_given_er_neg = pr_labels[(er_labels == 0)]

            er_pred_given_pr_pos = er_pred[(pr_labels == 1)]
            er_pred_given_pr_neg = er_pred[(pr_labels == 0)]
            pr_pred_given_er_pos = pr_pred[(er_labels == 1)]
            pr_pred_given_er_neg = pr_pred[(er_labels == 0)]

            batch_er_loss_given_pr_pos = _compute_subset_loss(
                er_pred_given_pr_pos, er_labels_given_pr_pos, ER_POS_GIVEN_PR_POS_PROB
            )
            batch_er_loss_given_pr_neg = _compute_subset_loss(
                er_pred_given_pr_neg, er_labels_given_pr_neg, ER_POS_GIVEN_PR_NEG_PROB
            )
            batch_pr_loss_given_er_pos = _compute_subset_loss(
                pr_pred_given_er_pos, pr_labels_given_er_pos, PR_POS_GIVEN_ER_POS_PROB
            )
            batch_pr_loss_given_er_neg = _compute_subset_loss(
                pr_pred_given_er_neg, pr_labels_given_er_neg, PR_POS_GIVEN_ER_NEG_PROB
            )

            batch_er_loss = (
                batch_er_loss_given_pr_pos + batch_er_loss_given_pr_neg
            ) / 2
            batch_pr_loss = (
                batch_pr_loss_given_er_pos + batch_pr_loss_given_er_neg
            ) / 2
        else:
            batch_er_loss = _compute_subset_loss(er_pred, er_labels, ER_POS_PROB)
            batch_pr_loss = _compute_subset_loss(pr_pred, pr_labels, PR_POS_PROB)

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
