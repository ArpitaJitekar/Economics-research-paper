import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Step 1: Load the dataset
data = pd.read_csv("C:\\Users\\Arpita\\Downloads\\archive (2)\\fer2013.csv")
print(data.head())

# Step 2: Convert the pixels column to images
def process_image(pixels):
    image = np.fromstring(pixels, dtype=int, sep=' ')
    image = image.reshape(48, 48)
    image = np.uint8(image)
    return image

data['image'] = data['pixels'].apply(process_image)

# Step 3: Normalize the images (scale between 0-1)
data['image'] = data['image'].apply(lambda img: img / 255.0)

# Step 4: Convert images to NumPy array
X = np.stack(data['image'].values)
y = data['emotion'].values

# Step 5: Reshape the images to fit CNN input (48x48x1)
X = X.reshape(-1, 48, 48, 1)

# Step 6: One-hot encode the labels (since it's a classification task)
y = to_categorical(y, num_classes=7)

# Step 7: Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build the VGG19 Model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Convert 1-channel grayscale to 3-channel grayscale for VGG19
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(7, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Step 9: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 10: Train the Model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=10, batch_size=64)

# Step 11: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Step 12: Plot the Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 13: Test with Random Images
plt.imshow(X_test[0].reshape(48, 48), cmap='gray')
pred = model.predict(X_test[0].reshape(1, 48, 48, 3))
emotion = np.argmax(pred)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(f'Predicted Emotion: {emotions[emotion]}')
