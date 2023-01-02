from typing import Union, Literal

TimeRange_t = Union[list, tuple, range]
PredictorsOp_t = Literal["mean", "std", "median"]
SpecialPredictor_t = tuple[str, TimeRange_t, PredictorsOp_t]