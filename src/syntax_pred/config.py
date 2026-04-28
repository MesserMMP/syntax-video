# src/syntax_pred/config.py — единая конфигурация и устройство

from omegaconf import OmegaConf
from .utils import pick_device

CFG = OmegaConf.load("configs/default.yaml")
DEVICE = pick_device(CFG.device)
