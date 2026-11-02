"""Adapted from qiskit-machine-learning (Apache-2.0)."""
from .loss_functions import Loss, L1Loss, L2Loss, CrossEntropyLoss
from .kernel_loss_functions import KernelLoss, SVCLoss, SVRLoss, MSRLoss, MARLoss, HuberLoss
__all__ = ["Loss", "L1Loss", "L2Loss", "CrossEntropyLoss", "KernelLoss", "SVCLoss", "SVRLoss", "MSRLoss", "MARLoss", "HuberLoss"]
