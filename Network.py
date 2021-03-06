import tensorflow as tf
import tensorflow_addons as tfa

layers = tf.keras.layers

# measuring triplet accuracy during training
def triplet_accuracy(y_true, y_pred):
    batch_size =tf.cast(tf.size(y_true), dtype=tf.dtypes.float32)
    # Build pairwise squared distance matrix
    pdist_matrix = pairwise_distance(y_pred, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = tf.cast(tf.math.equal(y_true, tf.transpose(y_true)), dtype=tf.dtypes.float32)
    # Invert so we can select negatives only.
    adjacency_not = 1-adjacency
    
    # Convert to decision with thresholding at 0.5
    predicted = tf.cast(tf.math.less_equal(pdist_matrix, 0.5), dtype=tf.dtypes.float32)
    
    # Calculate true positives and true negatives
    true_trues = tf.reduce_sum(tf.cast(
        tf.math.multiply(predicted, adjacency)
        , dtype=tf.dtypes.float32))
    true_falses = tf.reduce_sum(tf.cast(
        tf.math.multiply(1-predicted, adjacency_not)
        , dtype=tf.dtypes.float32))
    
    # Calculate percentage
    return (true_trues+true_falses)/(batch_size*batch_size)

# model definition
def create_custom_model(data_dim=(None, 204), filter_size=512, output_size=512):
    inputs = layers.Input(shape=(data_dim), name="input")

    x = layers.BatchNormalization(name='conv1_bn')(inputs)
    x = layers.Conv1D(filters=filter_size, kernel_size=3, name='conv1', padding="same")(x)
    x = layers.MaxPool1D(pool_size=2, padding="same", name='maxpool1')(x)
    
    x = layers.BatchNormalization(name='conv2_bn')(x)
    x = layers.Activation('relu', name='conv2_relu')(x)
    x = layers.Conv1D(filters=filter_size, kernel_size=3, name='conv2', padding="same")(x)
    x = layers.MaxPool1D(pool_size=2, padding="same", name='maxpool2')(x)
    
    x = layers.BatchNormalization(name='conv3_bn')(x)
    x = layers.Activation('relu', name='conv3_relu')(x)
    x = layers.Conv1D(filters=filter_size, kernel_size=3, name='conv3', padding="same")(x)
    
    x = layers.BatchNormalization(name='conv4_bn')(x)
    x = layers.Activation('relu', name='conv4_relu')(x)
    x = layers.Conv1D(filters=filter_size, kernel_size=3, name='conv4', padding="same")(x)
    
    convE = layers.BatchNormalization(name='convE_bn')(x)
    convE = layers.Activation('relu', name='convE_relu')(convE)
    convE = layers.Conv1D(filters=filter_size*3, kernel_size=1, name='convE', padding="same")(convE)

    tempavg = layers.GlobalAveragePooling1D(name="avg")(convE)
    tempstd = layers.Lambda(lambda x: K.sqrt(K.var(x, axis=1)+0.00001), name="std")(convE)

    avg_std = layers.Concatenate(name="concat")([tempavg, tempstd])

    dense1 = layers.BatchNormalization(name='bn_dense1')(avg_std)
    dense1 = layers.Dense(units=filter_size, name='dense1')(dense1)

    denseE = layers.BatchNormalization(name='bn_denseE')(dense1)
    denseE = layers.Activation('relu', name='relu_denseE')(denseE)
    denseE = layers.Dense(units=output_size, name='output')(denseE)
    output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="L2_norm")(denseE)

    model = tf.keras.Model(inputs, output, name="mynet")
    model.summary()
    return model

# training parameters
batchsize = 256
featdim = 512
data_dim=(None,3*lnc)
output_size = featdim

# model definition
model = create_custom_model(data_dim=data_dim, filter_size=featdim, output_size = output_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(.0001),
    loss=tfa.losses.TripletSemiHardLoss(), metrics=triplet_accuracy)
    
# training
log = model.fit(trainset, initial_epoch = 0, epochs=100, verbose=1, validation_data=validset)

# embedding extraction
enrollment_E = model.predict(enrollment_dataset, verbose=1)
test_E = model.predict(test_dataset, verbose=1)

# enrollment
workers = 12

enroll_ids = np.unique(enrollment_list[:,0])

enroll_label = []
enrollment = []
for eid in enroll_ids:
    ixs = np.where(eid==enrollment_list[:,0])[0]
    enrollment.append(np.mean(enrollment_E[ixs],axis=0))
    enroll_label.append(eid)
    
# test
# Get pairwise distance matrice
score_mat = sklearn.metrics.pairwise_distances(enrollment, test_E, n_jobs=workers)

# Score normalization
score_mat -= score_mat.mean(axis=0, keepdims=True)
score_mat /= score_mat.std(axis=0, keepdims=True)

score = []
for n, tid in enumerate(test_list[:,0]):
    score.append(score_mat[enroll_label.index(tid),n])

# softmax classifier
batchs = 64

# name to one-hot representation
def name_to_one_hot(name):
    ix = tf.where(enroll_ids==name)[0][0]
    return tf.one_hot(ix, len(enroll_ids))

# Training dataset
train_data = tf.data.Dataset.from_tensor_slices(enrollment_E)
train_labl = tf.data.Dataset.from_tensor_slices(enrollment_list[:,0]).map(name_to_one_hot)
trn_dataset = tf.data.Dataset.zip((train_data,train_labl)).shuffle(len(enrollment_E)).batch(batchs)

# Test dataset
test_data = tf.data.Dataset.from_tensor_slices(test_E[test_list[:,2]=='client'])
test_labl = tf.data.Dataset.from_tensor_slices(test_list[test_list[:,2]=='client',0]).map(name_to_one_hot)
tst_dataset = tf.data.Dataset.zip((test_data,test_labl)).batch(batchs)

# Softmax classifier parameters
filter_size = 512
inputs = 512
classes = len(enroll_ids)

# Model definition
inputs = layers.Input(shape=(512), name="input")
outputs = layers.Dense(units=classes, activation='softmax', name='output')(x)

model = tf.keras.Model(inputs, outputs, name="mynet")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train model
log = model.fit(trn_dataset, epochs=100, verbose=1, validation_data=tst_dataset,class_weight=cw_dict)

# Test softmax classifier
complete_test = tf.data.Dataset.from_tensor_slices(test_E).batch(batchs)

# Model prediction scores
score_mat = model_softmax.predict(complete_test, verbose=1)

# Convert to test-list utterances
score = []
for n, tid in enumerate(test_list[:,0]):
    score.append(score_mat[n, enroll_label.index(tid)])
