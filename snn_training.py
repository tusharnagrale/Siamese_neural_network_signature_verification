# Add these imports
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Replace the process_images function with a better approach
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
            
            # Add preprocessing to enhance signature features
            # Apply adaptive thresholding to better capture signature strokes
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label)
    return images, labels

# Process images with better splitting
org_images, org_labels = process_images(full_org_path, 0)
forg_images, forg_labels = process_images(full_forg_path, 1)

# Combine and shuffle all data
all_images = np.array(org_images + forg_images) / 255.0
all_labels = np.array(org_labels + forg_labels)

# Use proper stratified splitting
X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

print(f"Training data size: {X_train.shape}")
print(f"Training labels size: {y_train.shape}")
print(f"Testing data size: {X_test.shape}")
print(f"Testing labels size: {y_test.shape}")

# Improved pairing function with balanced classes
def create_balanced_pairs(X, y, num_pairs_per_class=1000):
    X_pairs, y_pairs = [], []
    
    # Get indices for each class
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    
    # Create positive pairs (same class)
    for _ in range(num_pairs_per_class):
        # Same class - both genuine
        idx1, idx2 = np.random.choice(class0_idx, 2, replace=False)
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(1)
        
        # Same class - both forged
        idx1, idx2 = np.random.choice(class1_idx, 2, replace=False)
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(1)
    
    # Create negative pairs (different classes)
    for _ in range(num_pairs_per_class * 2):  # Balance with positive pairs
        idx1 = np.random.choice(class0_idx, 1)[0]
        idx2 = np.random.choice(class1_idx, 1)[0]
        X_pairs.append([X[idx1], X[idx2]])
        y_pairs.append(0)
    
    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)
    
    # Shuffle the pairs
    X_pairs, y_pairs = shuffle(X_pairs, y_pairs, random_state=42)
    
    return X_pairs, y_pairs

# Create balanced pairs
train_pairs, train_pair_labels = create_balanced_pairs(X_train, y_train, num_pairs_per_class=500)
test_pairs, test_pair_labels = create_balanced_pairs(X_test, y_test, num_pairs_per_class=200)

print(f"Train pairs shape: {train_pairs.shape}")
print(f"Train pair labels shape: {train_pair_labels.shape}")
print(f"Test pairs shape: {test_pairs.shape}")
print(f"Test pair labels shape: {test_pair_labels.shape}")

# Improved model architecture with regularization
def get_cnn_block(depth, dropout_rate=0.3):
    return Sequential([
        Conv2D(depth, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(2),
        Dropout(dropout_rate)
    ])

DEPTH = 32
cnn = Sequential([
    Reshape((64, 64, 1)),
    get_cnn_block(DEPTH, 0.2),
    get_cnn_block(DEPTH*2, 0.3),
    get_cnn_block(DEPTH*4, 0.4),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
])

# Rest of the model definition remains the same...

# Add learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch > 20:
        return lr * 0.1
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Add more callbacks
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_loss', mode='min'
)

# Train for more epochs
history = model.fit(
    x=[train_pairs[:, 0, :, :], train_pairs[:, 1, :, :]],
    y=train_pair_labels,
    validation_data=(
        [test_pairs[:, 0, :, :], test_pairs[:, 1, :, :]],
        test_pair_labels
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, lr_callback],
    verbose=1
)

# Add evaluation metrics
def evaluate_model(model, test_pairs, test_pair_labels):
    y_pred = model.predict([test_pairs[:, 0, :, :], test_pairs[:, 1, :, :]])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(test_pair_labels, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Calculate optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold

optimal_threshold = evaluate_model(model, test_pairs, test_pair_labels)