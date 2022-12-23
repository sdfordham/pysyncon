library(tidyverse)
library(Synth)

load('smoking.rda')

fac <- factor(smoking$state)
smoking$state_no <- as.numeric(fac)

non.controls <- c('Alaska', 'Arizona', 'California', 'District of Columbia',
                  'Florida', 'Hawaii', 'Maryland', 'Massachusetts', 'Michigan',
                  'New Jersey', 'New York', 'Oregon', 'Washington')
control.states <- smoking %>%
  dplyr::filter(!state %in% non.controls)  %>%
  select(state) %>%
  unique()

message(smoking)
message(mode(smoking[,'state_no']))
dataprep(
  foo = smoking,
  predictors = c('cigsale', 'lnincome', 'beer', 'age15to24', 'retprice'),
  unit.variable = 'state_no',
  time.variable = 'year',
  treatment.identifier = 'California',
  controls.identifier - controls.states,
  unit.names.variable = 'state'
)
