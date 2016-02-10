import pandas
import numpy

type_map = dict(MCARR_ID=numpy.dtype(str))
vdf = pandas.read_csv('vehicle_train.csv', dtype=type_map)
