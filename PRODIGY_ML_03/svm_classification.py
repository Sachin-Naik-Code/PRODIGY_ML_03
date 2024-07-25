import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for label in ['cats', 'dogs']:
        dir_path = os.path.join(data_dir, label)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img.flatten())  # Flatten the image
            labels.append(0 if label == 'cats' else 1)
    return np.array(images), np.array(labels)

# Load training data
train_dir = 'C:\\Users\\sachin\\PRODIGY_ML_03\\train'
X_train, y_train = load_data(train_dir)

# Train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'svm_model.pkl')

# Load test data
test_dir = 'C:\\Users\\sachin\\PRODIGY_ML_03\\test'
X_test, y_test = load_data(test_dir)

# Load the model
model = joblib.load('svm_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
