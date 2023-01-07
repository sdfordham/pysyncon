from typing import Union, Literal


TimeRange_t = Union[list, range]
PredictorsOp_t = Literal["mean", "std", "median"]
SpecialPredictor_t = tuple[str, TimeRange_t, PredictorsOp_t]
OptimizerMethod_t = Literal[
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "trust-constr"
]
