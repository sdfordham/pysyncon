library(Synth)

load('smoking.rda')
df <- as.data.frame(smoking)

treated.states <- 'California'
control.states <-  c('Alabama', 'Arkansas', 'Colorado', 'Connecticut',
                     'Delaware', 'Georgia', 'Idaho', 'Illinois', 'Indiana',
                     'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
                     'Minnesota', 'Mississippi', 'Missouri', 'Montana',
                     'Nebraska', 'Nevada', 'New Hampshire', 'New Mexico',
                     'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
                     'Pennsylvania', 'Rhode Island', 'South Carolina',
                     'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
                     'Virginia', 'West Virginia', 'Wisconsin', 'Wyoming')

fac <- factor(df$state)
df$state.no <- as.numeric(fac)

dataprep.out <- dataprep(
  foo = df,
  predictors = c('lnincome', 'beer', 'age15to24', 'retprice'),
  predictors.op = 'mean',
  special.predictors = list(
    list(which(names(df) == 'lnincome'), 1980:1988, 'mean'),
    list(which(names(df) == 'retprice'), 1980:1988, 'mean'),
    list(which(names(df) == 'age15to24'), 1980:1988, 'mean'),
    list(which(names(df) == 'beer'), 1984:1988, 'mean'),
    list(which(names(df) == 'cigsale'), 1975, 'mean'),
    list(which(names(df) == 'cigsale'), 1980, 'mean'),
    list(which(names(df) == 'cigsale'), 1988, 'mean')
  ),
  dependent = 'cigsale',
  unit.variable = 'state.no',
  time.variable = 'year',
  treatment.identifier = which(levels(fac) == treated.states),
  controls.identifier = control.states,
  unit.names.variable = 'state',
  time.predictors.prior = c(1970:1988),
  time.optimize.ssr = c(1970:1988),
  time.plot = c(1970:2000)
)

synth.out <- synth(dataprep.out)

path.plot(
  dataprep.res = dataprep.out,
  synth.res = synth.out,
  tr.intake = 1988
)

state.weights <- setNames(control.states, synth.out$solution.w)
print(state.weights)