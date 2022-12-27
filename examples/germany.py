import pandas as pd
from ..pyaugsynth import dataprep


df = pd.read_csv("./examples/germany.csv")

out = dataprep(
    foo=df,
    predictors=("gdp", "trade", "infrate"),
    predictors_op="mean",
    time_predictors_prior=range(1971, 1981),
    special_predictors=(
        ("industry", range(1971, 1981), "mean"),
        ("schooling", [1970, 1975], "mean"),
        ("invest70", [1980], "mean"),
    ),
    dependent="gdp",
    unit_variable="country",
    time_variable="year",
    treatment_identifier="West Germany",
    controls_identifier=(
        "USA",
        "UK",
        "Austria",
        "Belgium",
        "Denmark",
        "France",
        "Italy",
        "Netherlands",
        "Norway",
        "Switzerland",
        "Japan",
        "Greece",
        "Portugal",
        "Spain",
        "Australia",
        "New Zealand",
    ),
    time_optimize_ssr=range(1981, 1991),
    time_plot=range(1960, 2004),
)
