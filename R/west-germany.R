require(Synth)

load('west-germany.rda')
df <- as.data.frame(germany)

treated.countries <- 'West Germany'
control.countries <- c('USA', 'UK', 'Austria', 'Belgium', 'Denmark', 'France',
                       'Italy', 'Netherlands', 'Norway', 'Switzerland', 'Japan',
                       'Greece', 'Portugal', 'Spain', 'Australia', 'New Zealand'
                       )

training.dataprep.out <- dataprep(
  foo = df,
  predictors = c('gdp', 'trade', 'infrate'),
  special.predictors = list(
    list('industry', 1971:1980, 'mean'),
    list('schooling', c(1970,1975), 'mean'),
    list('invest70', 1980, 'mean')
  ),
  dependent = 'gdp',
  unit.variable = 'code',
  unit.names.variable = 'country',
  time.variable = 'year',
  treatment.identifier = 7,
  controls.identifier = control.countries,
  time.predictors.prior = 1971:1980,
  time.optimize.ssr = 1981:1990,
  time.plot = 1960:2003
)

training.synth.out <- synth(
  data.prep.obj=training.dataprep.out,
  Margin.ipop=.005,
  Sigf.ipop=7,
  Bound.ipop=6
)

dataprep.out <- dataprep(
  foo = df,
  predictors = c('gdp', 'trade', 'infrate'),
  predictors.op = 'mean',
  special.predictors = list(
    list('industry', 1981:1990, 'mean'),
    list('schooling', c(1980, 1985), 'mean'),
    list('invest80', 1980, 'mean')
  ),
  dependent = 'gdp',
  unit.variable = 'code',
  unit.names.variable = 'country',
  time.variable = 'year',
  treatment.identifier = 7,
  controls.identifier = control.countries,
  time.predictors.prior = 1981:1990,
  time.optimize.ssr = 1960:1989,
  time.plot = 1960:2003
)

synth.out <- synth(
  dataprep.out,
  custom.v=as.numeric(training.synth.out$solution.v)
)

path.plot(
  dataprep.res = dataprep.out,
  synth.res = synth.out,
  tr.intake = 1990
)

state.weights <- setNames(control.countries, round(synth.out$solution.w, digits=2))
print(state.weights)