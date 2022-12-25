require(Synth)

data(basque)

treated.regions <- 'Basque Country (Pais Vasco)'
control.regions <- c('Spain (Espana)', 'Andalucia', 'Aragon',
                     'Principado De Asturias', 'Baleares (Islas)', 'Canarias',                   
                     'Cantabria', 'Castilla Y Leon', 'Castilla-La Mancha',
                     'Cataluna', 'Comunidad Valenciana', 'Extremadura',
                     'Galicia', 'Madrid (Comunidad De)', 'Murcia (Region de)',
                     'Navarra (Comunidad Foral De)', 'Rioja (La)')

dataprep.out <- dataprep(
  foo = basque,
  predictors = c('school.illit', 'school.prim', 'school.med',
                 'school.high', 'school.post.high', 'invest'),
  predictors.op = 'mean',
  time.predictors.prior = 1964:1969,
  special.predictors = list(
    list('gdpcap', 1960:1969 ,'mean'),
    list('sec.agriculture', seq(1961, 1969, 2), 'mean'),
    list('sec.energy', seq(1961, 1969, 2), 'mean'),
    list('sec.industry', seq(1961, 1969, 2), 'mean'),
    list('sec.construction', seq(1961, 1969, 2), 'mean'),
    list('sec.services.venta', seq(1961, 1969, 2), 'mean'),
    list('sec.services.nonventa', seq(1961, 1969, 2), 'mean'),
    list('popdens', 1969, 'mean')
  ),
  dependent = 'gdpcap',
  unit.variable = 'regionno',
  unit.names.variable = 'regionname',
  time.variable = 'year',
  treatment.identifier = 17,
  controls.identifier = control.regions,
  time.optimize.ssr = 1960:1969,
  time.plot = 1955:1997
)

synth.out <- synth(
  data.prep.obj = dataprep.out,
  method = 'BFGS'
)

path.plot(
  dataprep.res = dataprep.out,
  synth.res = synth.out,
  tr.intake = 1998
)

state.weights <- setNames(
  control.regions,
  round(synth.out$solution.w, digits = 2)
)
print(state.weights)