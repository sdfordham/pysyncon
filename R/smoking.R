library(tidyverse)
library(Synth)

load('smoking.rda')
df <- as.data.frame(smoking)

treated.states <- 'California'
control.states <-  c('Rhode Island', 'Tennessee', 'Indiana', 'Nevada', 'Louisiana',
              'Oklahoma', 'New Hampshire', 'North Dakota', 'Arkansas',
              'Virginia', 'Illinois', 'South Dakota', 'Utah', 'Georgia',
              'Mississippi', 'Colorado', 'Minnesota', 'Texas', 'Kentucky',
              'Maine', 'North Carolina', 'Montana', 'Vermont', 'Iowa',
              'Connecticut', 'Kansas', 'Delaware', 'Wisconsin', 'Idaho',
              'New Mexico', 'West Virginia', 'Pennsylvania', 'South Carolina',
              'Ohio', 'Nebraska')
ignore.states <- c('Alaska', 'Arizona', 'California', 'District of Columbia',
            'Florida', 'Hawaii', 'Maryland', 'Massachusetts', 'Michigan',
            'New Jersey', 'New York', 'Oregon', 'Washington')

fac <- factor(df$state)
df$state_no <- as.numeric(fac)
df <- df %>% dplyr::filter(state == treated | state %in% control)

dataprep(
  foo = df,
  predictors = c('lnincome', 'beer', 'age15to24', 'retprice'),
  predictors.op = 'mean',
  dependent = 'cigsale',
  unit.variable = 'state_no',
  time.variable = 'year',
  treatment.identifier = which(levels(fac) == 'California'),
  controls.identifier = control.states,
  unit.names.variable = 'state',
  time.predictors.prior = c(1984:1989),
  time.optimize.ssr = c(1984:1990)
)
