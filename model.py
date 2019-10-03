from __future__ import absolute_import, division, print_function, unicode_literals

import csv
from datetime import datetime as dt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pandas import DataFrame

######################################
## Reading and preprocessing data.  ##
######################################

# # Reading main table which contains the customerID we need to 
# # get the consumption per client and the building class we will 
# # use to train the model
# with open('CustomersEspoo.csv', 'r') as customerFile:
#     reader = csv.DictReader(customerFile, delimiter=';')
#     data = [r for r in reader]

# # Deleting rows which can not be used to train the model.
# customerData = []
# for row in data:
#     if row['CLASS'] != 'NaN':
#         customerData.append({ 
#             'CUSTOMERID': row['CUSTOMERID'],
#             'CLASS': row['CLASS']
#         })

# # Reading consumption per client.
# for row in customerData:
#     with open('./ConsumptionData/'+row['CUSTOMERID']+'_consumption_kw.csv', 'r') as consumptionFile:
#         reader = csv.DictReader(consumptionFile, delimiter=';')
#         consumption = [r for r in reader]
#         row['DAYS'] = []
#         lastDay = None
#         countingDays = -1
#         for hour in consumption:
#             day = dt.strptime(hour['UTC_TIMESTAMP'],'%Y-%m-%d %H:%M:%S').date()
#             if day == lastDay:
#                 row['DAYS'][countingDays][str(day)].append(hour['CONSUMPTION_KW'])
#             else:
#                 row['DAYS'].append({
#                     str(day): [hour['CONSUMPTION_KW']] 
#                 })
#                 lastDay = day
#                 countingDays += 1

# # We save the data and restore it at this point using pickle:
# # Saving customerData with the information necessary, and ready to preprocess.
# with open('customerData.pkl', 'wb') as f:
#     pickle.dump([customerData], f)
# # Getting back customerData and use it to get a dataset
with open('customerData.pkl', 'rb') as f:
    customerData = pickle.load(f)[0]

# We do not know how many classes are contained on the files, so 
# we iterate through customerData to obtain all the classes.
classes = []
for row in customerData:
    if row['CLASS'] not in classes:
        classes.append(row['CLASS'])

print("Classes: " + str(classes))
print("Total number of classes: " + str(len(classes)))

########################
## Creating dataframe ##
########################

columns = ['FEATURE1','FEATURE2','FEATURE3','FEATURE4',
           'FEATURE5','FEATURE6','FEATURE7','FEATURE8',
           'FEATURE9','FEATURE10','FEATURE11','FEATURE12',
           'FEATURE13','FEATURE14','FEATURE15','FEATURE16',
           'FEATURE17','FEATURE18','FEATURE19','FEATURE20',
           'FEATURE21','FEATURE22','FEATURE23','FEATURE24',
           'FEATUREDAY', 'CLASS']

# Initialize the set of data
pdData = {
    'FEATURE1': [],
    'FEATURE2': [],
    'FEATURE3': [],
    'FEATURE4': [],
    'FEATURE5': [],
    'FEATURE6': [],
    'FEATURE7': [],
    'FEATURE8': [],
    'FEATURE9': [],
    'FEATURE10': [],
    'FEATURE11': [],
    'FEATURE12': [],
    'FEATURE13': [],
    'FEATURE14': [],
    'FEATURE15': [],
    'FEATURE16': [],
    'FEATURE17': [],
    'FEATURE18': [],
    'FEATURE19': [],
    'FEATURE20': [],
    'FEATURE21': [],
    'FEATURE22': [],
    'FEATURE23': [],
    'FEATURE24': [],
    'FEATUREDAY': [],
    'CLASS': []
}
for row in customerData:
    # Customer
    for d in row['DAYS']:
        day = list(d.keys())[0]
        counter = 1
        for hour in d[day]:
            pdData['FEATURE'+str(counter)].append(hour)
            counter += 1
        pdData['FEATUREDAY'].append(dt.strptime(day,'%Y-%m-%d').weekday() + 1)
        pdData['CLASS'].append(row['CLASS'])


df = DataFrame( pdData, columns = columns )

######### NOTE:
# After this point we were running out of time,
# so, we concentrate on solving this specific problem
# without generalize the code - but it is possible to do it.

# We need to give numbers to our model:
target = []
for c in df['CLASS']:
    if c == 'outdoorligths':
        target.append(0)
    elif c == 'trafficlights':
        target.append(1)
    elif c == 'kindergarten':
        target.append(2)
    elif c == 'playgroundwithbuilding':
        target.append(3)
    elif c == 'nursinghome':
        target.append(4)
    elif c == 'school':
        target.append(5)
    elif c == 'healthcenter':
        target.append(6)
    elif c == 'officebuilding':
        target.append(7)
    elif c == 'indoorswimmingpool':
        target.append(8)
    elif c == 'hospital':
        target.append(9)
    elif c == 'firestation':
        target.append(10)
    elif c == 'watertower':
        target.append(11)
    elif c == 'library':
        target.append(12)
    elif c == 'sportcenter':
        target.append(13)

# Classes: ['outdoorligths', 'trafficlights', 'kindergarten', 'playgroundwithbuilding', 'n
# ursinghome', 'school', 'healthcenter', 'officebuilding', 'indoorswimmingpool', 'hospital
# ', 'firestation', 'watertower', 'library', 'sportcenter']
# Total number of classes: 14

# We remove the column with class names and add the column with class numbers
df['TARGET'] = target
df.pop('CLASS')


#########################################
##  Building the model                 ##
#########################################

model = keras.Sequential([
    keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    
    # We could generalize the number of nodes of the output layer
    # according to the number of classes our data contains.
    keras.layers.Dense(14, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
            # We use sparse cat. crossentropy because my targets are integers.
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

###############################
## Training the model        ##
###############################

# Split the data into train and test
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

# Split features from labels
train_labels = train_dataset.pop('TARGET')
test_labels = test_dataset.pop('TARGET')

model.fit(
    tf.convert_to_tensor(train_dataset.values, tf.float32),
    tf.convert_to_tensor(train_labels.values, tf.int32), 
    epochs = 15, validation_split = 0.2, verbose = 0)

#####################################
## Evaluating the model            ##
#####################################

test_loss, test_acc = model.evaluate(
    tf.convert_to_tensor(test_dataset.values, tf.float32),
    tf.convert_to_tensor(test_labels.values, tf.int32)
)
print('Test accuracy: ', test_acc)

###
# NOTE: To evaluate precisely are model, we should train it 
# enough times when we use random rows from our data