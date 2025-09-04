# ================= IMPORTS =================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Use tf.keras consistently
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models, callbacks


# ================= IMAGE PREPROCESSING =================
def process_images(folder_path, label):
    images = []
    labels = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.endswith(('.jpeg', '.JPEG', '.png', '.PNG', '.jpg', '.JPG')):
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Adaptive thresholding for better signature stroke capture
            image = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label)
    return images, labels


# ================= LOAD DATA =================
full_forg_path = "signatures/full_forg"
full_org_path = "signatures/full_org"

org_images, org_labels = process_images(full_org_path, 0)
forg_images, forg_labels = process_images(full_forg_path, 1)

all_images = np.array(org_images + forg_images) / 255.0
all_labels = np.array(org_labels + forg_labels)

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")


# ================= PAIR GENERATION =================
def create_balanced_pairs(X, y, num_pairs_per_class=1000):
    X_pairs, y_pairs = [], []

    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]

    for _ in range(num_pairs_per_class):
        idx1, idx2 = np.random.choice(class0_idx, 2, replace=False)
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(1)

        idx1, idx2 = np.random.choice(class1_idx, 2, replace=False)
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(1)

    for _ in range(num_pairs_per_class * 2):
        idx1 = np.random.choice(class0_idx, 1)[0]
        idx2 = np.random.choice(class1_idx, 1)[0]
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(0)

    return shuffle(np.array(X_pairs), np.array(y_pairs), random_state=42)


train_pairs, train_pair_labels = create_balanced_pairs(X_train, y_train, 500)
test_pairs, test_pair_labels = create_balanced_pairs(X_test, y_test, 200)

print(f"Train pairs shape: {train_pairs.shape}")
print(f"Test pairs shape: {test_pairs.shape}")


# ================= CNN FEATURE EXTRACTOR =================
def get_cnn_block(depth, dropout_rate=0.3):
    return models.Sequential([
        layers.Conv2D(depth, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Dropout(dropout_rate)
    ])


DEPTH = 32
cnn = models.Sequential([
    layers.Reshape((64, 64, 1)),
    get_cnn_block(DEPTH, 0.2),
    get_cnn_block(DEPTH * 2, 0.3),
    get_cnn_block(DEPTH * 4, 0.4),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
])


# ================= SIAMESE MODEL =================
def l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

input_a = layers.Input(shape=(64, 64))
input_b = layers.Input(shape=(64, 64))

feat_a = cnn(input_a)
feat_b = cnn(input_b)

distance = layers.Lambda(l1_distance, name="l1_distance")([feat_a, feat_b])
output = layers.Dense(1, activation='sigmoid')(distance)

model = models.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# ================= CALLBACKS =================
def lr_scheduler(epoch, lr):
    return lr * 0.1 if epoch > 20 else lr

lr_callback = callbacks.LearningRateScheduler(lr_scheduler)
early_stopping = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_loss', mode='min'
)


# ================= TRAINING =================
history = model.fit(
    x=[train_pairs[:, 0, :, :], train_pairs[:, 1, :, :]],
    y=train_pair_labels,
    validation_data=([test_pairs[:, 0, :, :], test_pairs[:, 1, :, :]], test_pair_labels),
    epochs=40,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, lr_callback],
    verbose=1
)


# ================= EVALUATION =================
def evaluate_model(model, test_pairs, test_pair_labels):
    y_pred = model.predict([test_pairs[:, 0, :, :], test_pairs[:, 1, :, :]])

    fpr, tpr, thresholds = roc_curve(test_pair_labels, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    cm = confusion_matrix(test_pair_labels, (y_pred > optimal_threshold).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    print(classification_report(test_pair_labels, (y_pred > optimal_threshold).astype(int)))
    return optimal_threshold


optimal_threshold = evaluate_model(model, test_pairs, test_pair_labels)
