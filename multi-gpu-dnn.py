# You must have tensorflow-gpu
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Enable GPU Growth for all GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Use TF MirroredStrategy API to distribute training across multiple GPUs
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Run your model inside
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    print(x_train.shape)
    print(x_train[0].shape)

    model = Sequential()

    model.add(LSTM(128, input_shape=(
        x_train.shape[1:]),  return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    model.fit(x_train,
              y_train,
              epochs=3,
              validation_data=(x_test, y_test))
