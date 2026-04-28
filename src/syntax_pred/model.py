# src/syntax_pred/model.py — removed sigma & yulie, robust auto-loading, backward-compatible **kwargs
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tvmv
import lightning.pytorch as pl

class SyntaxLightningModule(pl.LightningModule):
    """
    R3D-18 backbone с настраиваемой "головой" для последовательностей.
    На выходе тензор (B, 2): [логит классификации, регрессия = log(1 + score)].
    """
    def __init__(
        self,
        num_classes: int,
        lr: float,
        variant: str,
        weight_decay: float = 0.0,
        max_epochs: Optional[int] = None,
        weight_path: Optional[str] = None,     # загрузка только бэкбона (опционально)
        pl_weight_path: Optional[str] = None,  # загрузка целого модуля (ckpt/pt)
        rnn_hidden_div: int = 4,
        rnn_dropout: float = 0.2,
        bert_nhead: int = 4,
        bert_layers: int = 1,
        bert_ff_div: int = 4,
        bert_dropout: float = 0.2,
        precision: str = "bf16",
        **_ignore,  # игнорирование лишних/неиспользуемых аргументов в вызове
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.variant = variant
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # --- Backbone (R3D-18)
        self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2, bias=True)

        if weight_path is not None:
            self._load_backbone_weight(weight_path, self.model)

        # Для последовательных голов используем признаки до FC
        if self.variant != "mean_out":
            self.model.fc = nn.Identity()

        # --- Heads
        if self.variant == "mean_out":
            pass
        elif self.variant in ("gru_mean", "gru_last"):
            self.rnn = nn.GRU(in_features, in_features // rnn_hidden_div, batch_first=True)
            self.dropout = nn.Dropout(rnn_dropout)
            self.fc = nn.Linear(in_features // rnn_hidden_div, num_classes)
        elif self.variant in ("lstm_mean", "lstm_last"):
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=in_features // rnn_hidden_div,
                proj_size=num_classes,
                batch_first=True,
            )
        elif self.variant == "mean":
            self.fc = nn.Linear(in_features, num_classes)
        elif self.variant in ("bert_mean", "bert_cls", "bert_cls2"):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=in_features,
                nhead=bert_nhead,
                batch_first=True,
                dim_feedforward=in_features // bert_ff_div,
                dropout=bert_dropout,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=bert_layers)
            self.dropout = nn.Dropout(bert_dropout)
            self.fc = nn.Linear(in_features, num_classes)
            if self.variant == "bert_cls2":
                self.cls = nn.Parameter(torch.randn(1, 1, in_features))
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Загрузка полного state_dict (поддерживает форматы ckpt/pt)
        if pl_weight_path is not None:
            self._load_full_module(pl_weight_path)

        # Лоссы
        self.loss_clf = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_reg = nn.MSELoss(reduction="none")

        # Кэши для валидационных метрик
        self.y_val: List[int] = []
        self.p_val: List[float] = []
        self.r_val: List[int] = []
        self.ty_val: List[float] = []
        self.tp_val: List[float] = []

    # ---------------- weights -----------------
    @staticmethod
    def _strip_prefix(key: str) -> str:
        for pref in ("model.", "backbone.", "module.", "net."):
            if key.startswith(pref):
                return key[len(pref):]
        return key

    def _load_backbone_weight(self, weight_path: str, model: nn.Module) -> None:
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        new_sd = {
            self._strip_prefix(k).replace("fc.", ""): v
            for k, v in ckpt.items()
            if k.startswith(("model.", "backbone.", "module."))
        }
        model.load_state_dict(new_sd, strict=False)

    def _load_block(self, sd: Dict[str, torch.Tensor], module: nn.Module, prefix: str) -> None:
        filtered = {k.replace(f"{prefix}.", ""): v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[load] Missing for {prefix}: {missing}")
        if unexpected:
            print(f"[load] Unexpected for {prefix}: {unexpected}")

    def _load_full_module(self, path: str) -> None:
        raw = torch.load(path, map_location="cpu")
        sd: Dict[str, torch.Tensor] = raw["state_dict"] if (isinstance(raw, dict) and "state_dict" in raw) else raw
        self._load_block(sd, self.model, "model")
        if self.variant == "mean_out":
            pass
        elif self.variant in ("gru_mean", "gru_last"):
            self._load_block(sd, self.rnn, "rnn")
            self._load_block(sd, self.fc, "fc")
        elif self.variant in ("lstm_mean", "lstm_last"):
            self._load_block(sd, self.lstm, "lstm")
        elif self.variant == "mean":
            self._load_block(sd, self.fc, "fc")
        elif self.variant in ("bert_mean", "bert_cls", "bert_cls2"):
            self._load_block(sd, self.encoder, "encoder")
            self._load_block(sd, self.fc, "fc")
            if self.variant == "bert_cls2" and "cls" in sd:
                with torch.no_grad():
                    self.cls.copy_(sd["cls"])

    # ---------------- forward -----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Подача батча последовательностей формы (B,S,C,T,H,W),
        где S — число клипов внутри исследования для одной артерии.
        """
        b, s = x.shape[:2]
        x = x.flatten(0, 1)   # (B*S, C, T, H, W)
        x = self.model(x)     # (B*S, 2) или (B*S, F) при fc=Identity
        x = x.unflatten(0, (b, s))  # (B, S, ...)

        if self.variant == "mean_out":
            x = x.mean(dim=1)
        elif self.variant in ("gru_mean", "gru_last"):
            all_outs, last = self.rnn(x)
            x = all_outs.mean(dim=1) if self.variant == "gru_mean" else last[0]
            x = self.dropout(x)
            x = self.fc(x)
        elif self.variant in ("lstm_mean", "lstm_last"):
            all_outs, (last_out, _) = self.lstm(x)
            x = all_outs.mean(dim=1) if self.variant == "lstm_mean" else last_out
        elif self.variant == "mean":
            x = x.mean(dim=1)
            x = self.fc(x)
        elif self.variant in ("bert_mean", "bert_cls", "bert_cls2"):
            if self.variant == "bert_cls":
                x = F.pad(x, (0, 0, 1, 0), value=0.0)  # добавление CLS токена
            elif self.variant == "bert_cls2":
                bs = x.size(0)
                x = torch.cat([self.cls.expand(bs, -1, -1), x], dim=1)
            x = self.encoder(x)
            x = x.mean(dim=1) if self.variant == "bert_mean" else x[:, 0, :]
            x = self.dropout(x)
            x = self.fc(x)
        else:
            raise ValueError(self.variant)
        return x  # (B, 2)

    # --------------- training (kept for completeness) ---------------
    def training_step(self, batch, batch_idx):
        x, y, target, _ = batch
        y_hat = self(x)
        yp_clf = y_hat[:, 0:1]
        yp_reg = y_hat[:, 1:]

        weights_clf = torch.where(y > 0, 1.0, 0.2)
        clf_loss = (self.loss_clf(yp_clf, y) * weights_clf).mean()

        reg_loss = self.loss_reg(yp_reg, target).mean()
        return clf_loss + 0.5 * reg_loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs:
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.max_epochs)
            return [opt], [sch]
        return opt
