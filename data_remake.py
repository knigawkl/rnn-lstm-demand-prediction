import datetime
import h5py
import numpy as np
from fuel.datasets import H5PYDataset

from config import config

locals().update(config)
network_mode = 0

file = h5py.File('input_double_20.hdf5', 'r')
input = np.array(file['features'], dtype='float32')
target = np.array(file['targets'], dtype='float32')
uniqueGeoHash = np.array(file['uniqueGeo'], dtype='S7')

wea_file = h5py.File('weather.hdf5', 'r')
weather = np.array(wea_file['wea'], dtype='float32')
print(weather[:10])

in_size = 2 * uniqueGeoHash.shape[0]

input = input[:, :, :in_size]
target = target[:, :, :uniqueGeoHash.shape[0]]
input = input.round()
target = target.round()

print(input[0, 1, :10])
print(target[0, 0, :10])

N1 = input.shape[0]
N2 = input.shape[1]
N3 = input.shape[2]

inputs = np.ones((N1, N2, N3 + 79), dtype='float32')
inputs[:, :, :N3] = input

startDateTime = datetime.datetime(2014, 1, 1, 0, 0, 0)
endDateTime = datetime.datetime(2014, 1, 1, 0, 20, 0)
sampleDataTime = endDateTime - startDateTime

index = 0
for i in range(int(N1)):
    for j in range(N2):
        currentDataTime = startDateTime + index * sampleDataTime
        index = index + 1
        time_info_each_step = np.zeros(67, dtype='float32')
        time_info_each_step[currentDataTime.month] = 1.0  # month
        time_info_each_step[currentDataTime.day] = 1.0  # day
        time_info_each_step[31 + currentDataTime.hour + 1] = 1.0  # hour
        time_info_each_step[55 + int(currentDataTime.minute / 20) + 1] = 1.0  # min
        time_info_each_step[(index / (3 * 24) + 2) % 7 + 1 + 59] = 1.
        inputs[i, j, N3:N3 + 67] = time_info_each_step
        inputs[i, j, N3 + 67:] = weather[index / (3 * 24)]

helper = np.array(inputs)
print('the std of pickup is: ' + str(np.std(helper)))
helper = helper[helper > 0]
pickups_std = np.ceil(2 * np.std(helper))
print('the max of pickup is: ' + str(np.max(helper)))
print('the std of pickup is: ' + str(pickups_std))

print(uniqueGeoHash[:10])

inputs[:, :, :in_size] = inputs[:, :, :in_size] / 10.
target[:, :, :uniqueGeoHash.shape[0]] = target[:, :, :uniqueGeoHash.shape[0]] / 10.
print(inputs[0, 1, :10])
print(target[0, 0, :10])

print('inputs shape:', inputs.shape)
print('outputs shape:', target.shape)
print('uniqueGeoHash shape:', uniqueGeoHash.shape)

f = h5py.File(hdf5_file[network_mode], mode='w')
features = f.create_dataset('features', inputs.shape, dtype='float32')
targets = f.create_dataset('targets', target.shape, dtype='float32')
uniqueGeo = f.create_dataset('uniqueGeo', uniqueGeoHash.shape, dtype="S7")

features[...] = inputs
targets[...] = target
uniqueGeo[...] = uniqueGeoHash
features.dims[0].label = 'batch'
features.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'
print(uniqueGeo[:10])
print(features[-1, -1, :10])
print(targets[-1, -2, :10])

nsamples = inputs.shape[0]
print(nsamples)
nsamples_train = int(nsamples * train_size[network_mode])

split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'test': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
