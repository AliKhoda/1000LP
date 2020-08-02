import tensorflow as tf
import tensorflow_addons as tfa

layers = tf.keras.layers

def triplet_accuracy(y_true, y_pred):
    batch_size =tf.cast(tf.size(y_true), dtype=tf.dtypes.float32)
    # Build pairwise squared distance matrix
    pdist_matrix = pairwise_distance(y_pred, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = tf.cast(tf.math.equal(y_true, tf.transpose(y_true)), dtype=tf.dtypes.float32)
    # Invert so we can select negatives only.
    adjacency_not = 1-adjacency
    
    predicted = tf.cast(tf.math.less_equal(pdist_matrix, 0.5), dtype=tf.dtypes.float32)
    
    true_trues = tf.reduce_sum(tf.cast(
        tf.math.multiply(predicted, adjacency)
        , dtype=tf.dtypes.float32))
    true_falses = tf.reduce_sum(tf.cast(
        tf.math.multiply(1-predicted, adjacency_not)
        , dtype=tf.dtypes.float32))
    
    return (true_trues+true_falses)/(batch_size*batch_size)

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

batchsize = 256
featdim = 512
data_dim=(None,3*lnc)
output_size = featdim

model = create_custom_model(data_dim=data_dim, filter_size=featdim, output_size = output_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(.0001),
    loss=tfa.losses.TripletSemiHardLoss(), metrics=triplet_accuracy)
    
log = model.fit(trainset, initial_epoch = 0, epochs=100, verbose=1, validation_data=validset)
