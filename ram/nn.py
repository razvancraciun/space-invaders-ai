from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def init_model(input_dim, output_dim, learning_rate, loss):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu'),
        Dense(64, activation = 'relu'),
        Dense(output_dim)
    ])
    optimizer = Adam(lr = learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    return model