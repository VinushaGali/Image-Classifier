import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from google.colab import drive
drive.mount('/content/drive')
import time

# Define hyperparameters
NUM_LAYERS = 2
NUM_NEURONS = [32, 64]  # Number of neurons in each layer
CONV_SHAPE = (3, 3)  # Shape of convolutional layers

# Set up image data generator
# referred https://stackoverflow.com/questions/67240420/not-able-to-use-the-model-fit-in-keras for the below code.
TRAIN_DIR = '/content/drive/MyDrive/images/train'
VAL_DIR = '/content/drive/MyDrive/images/validation'
BATCH_SIZE = 32
IMG_SIZE = (48, 48)  # Size of input images
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')

# Build CNN model
# referred https://faroit.com/keras-docs/2.0.2/getting-started/sequential-model-guide/ and https://github.com/jaydeepthik/kaggle-facial-expression-recognition/blob/master/facial_expression.py to understand the sequential model.
model = Sequential()
model.add(Conv2D(NUM_NEURONS[0], CONV_SHAPE, activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(NUM_NEURONS[1], CONV_SHAPE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

start_time = time.time()

# Compile and fit model
# referred https://faroit.com/keras-docs/2.0.2/getting-started/sequential-model-guide/ for Compilation code.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 25
history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

end_time = time.time()
elapsed_time = end_time - start_time

# Evaluate model
model.evaluate(validation_generator)

# Collect accuracy values for each epoch from the history object
accuracy_values = history.history['accuracy']

# Calculate the average accuracy over all epochs
avg_accuracy = sum(accuracy_values) / len(accuracy_values)

# Print the average accuracy
print("Average accuracy: {:.2f}%".format(avg_accuracy * 100))

print('Training time:', elapsed_time)