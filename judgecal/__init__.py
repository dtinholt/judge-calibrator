"""judgecal: measure how consistent model judges are with themselves and with each other.

It reports test-retest agreement, pairwise Cohen's kappa, confusion tables, the share of
low-confidence verdicts and a judge-noise band to print beside a judged benchmark number.
"""
from .calibration import Agreement, Calibration, CalibrationResult, ClassAgreement, JudgeCost, JudgeSummary, is_low
from .compare import DriftResult, compare
from .items import Item, Items, load_strata
from .judges import Judge, ChatCompletionsJudge, JudgeError, ReplayJudge, Verdict
from .rubric import Rubric
from .stats import bootstrap_kappa, confusion, kappa, noise_band, specific_agreement, wilson

__version__ = "0.2.1"
__all__ = [
    "Agreement", "Calibration", "CalibrationResult", "ClassAgreement", "DriftResult", "JudgeCost",
    "JudgeSummary", "bootstrap_kappa", "compare", "is_low", "load_strata", "specific_agreement",
    "ChatCompletionsJudge", "Item", "Items", "Judge", "JudgeError", "ReplayJudge", "Verdict",
    "Rubric", "confusion", "kappa", "noise_band", "wilson",
]
