"""World model: PatchTST encoder, JEPA + SIGReg, probes, and training."""

from lewm_harness.world_model.encoder import PatchTSTEncoder, SequenceEncoder
from lewm_harness.world_model.model import FinancialLeWM
from lewm_harness.world_model.probes import Gate1Report, evaluate_gate1
from lewm_harness.world_model.sigreg import jepa_sigreg_loss, sigreg

__all__ = [
    "FinancialLeWM",
    "Gate1Report",
    "PatchTSTEncoder",
    "SequenceEncoder",
    "evaluate_gate1",
    "jepa_sigreg_loss",
    "sigreg",
]
