import keras
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from read_files import list_people
import numpy as np
from process_embeddings import trim_emb_matrix
import env, pickle
# import tqdm 


def generator(mat, people, encoder, batch_size=32, sh = True):
    if sh:
        shuffle = np.random.choice(len(mat), len(mat), replace = False)   
        mat = np.array(mat)[shuffle]

    while True: 
        X_train = []
        y_train = []
        start = 0
        end = batch_size
        while end < len(mat):
            res = np.array(mat)[start:end]
            X_train = np.array([x[1] for x in res])
            y_train = np.array([people[x[0]] for x in res])
            y_train = encoder.transform(y_train)
            start += batch_size
            end += batch_size
            yield X_train.reshape(-1,5, 2048, 1), to_categorical(y_train, num_classes=10)


batch_size = 128
num_classes = 10
epochs = 100
filter_pixel=5
noise = 1
droprate=0.25
test_train = 'train'

with open('beka.pickle', 'rb') as f:
        mat = pickle.load(f)
# data input dimensions
n_rows, n_cols = 5, 2048

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder().fit([p['category'] for p in people])
docs = load_files(env.learning_data_path, test_train)
people = list_people(docs)
people_dict = {}
for p in people:
    people_dict[p['name']] = p['category']

## need to be added
people_dict['Donald Tuska'] = 'politycy'

#Start Neural Network
input_shape = (n_rows, n_cols, 1)

model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 10), padding="same",
                 activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(droprate))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',border_mode="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(droprate))
encoder.transform(['lekarze'])

model.add(Flatten()) 
model.add(Dense(300, use_bias=False)) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dropout(droprate))   

model.add(Dense(10)) 
model.add(Activation('softmax')) 

gen = generator(mat, people_dict, encoder)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.fit_generator(gen ,epochs = 10, steps_per_epoch = 30)
model.summary()

with open('data.pickle', 'rb') as f:
    mat_test = pickle.load(f)

x = np.array([el[1] for el in mat_test])
x = x.reshape(-1,5, 2048, 1)
y = [people_dict[el[0]] for el in mat_test]
y = encoder.transform(y)
step = 5
ans = []
# predict in batches
for i in range(0, x.shape[0],step):
    ans.extend(model.predict(x[i:i+step,:,:,:]))

ans = np.array(ans).argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y, ans))
print(confusion_matrix(y, ans))


