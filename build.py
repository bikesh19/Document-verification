import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)  # standard size for CNNs
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # freeze pretrained layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # start small
)

model.save("model.h5")



# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# # -------------------------
# # Parameters
# # -------------------------
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# EPOCHS = 20
# DATASET_PATH = "dataset"  # must contain license/ and not_license/

# # -------------------------
# # Data Augmentation
# # -------------------------
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     zoom_range=0.2,
#     horizontal_flip=False,
#     fill_mode="nearest",
#     validation_split=0.2
# )

# train_data = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="training"
# )

# val_data = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="validation"
# )

# # -------------------------
# # Build the model (MobileNetV2)
# # -------------------------
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224,224,3),
#     include_top=False,
#     weights="imagenet"
# )
# base_model.trainable = False  # freeze for initial training

# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dropout(0.3),       # prevents overfitting
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# # -------------------------
# # Callbacks
# # -------------------------
# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
#     ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
# ]

# # -------------------------
# # Train the model
# # -------------------------
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=EPOCHS,
#     callbacks=callbacks
# )

# # -------------------------
# # Fine-tuning (optional)
# # -------------------------
# # Unfreeze last layers of MobileNetV2
# base_model.trainable = True
# for layer in base_model.layers[:-50]:  # freeze first layers, fine-tune last 50
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# history_ft = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10,
#     callbacks=callbacks
# )

# # -------------------------
# # Save final model
# # -------------------------
# model.save("final_model.keras")
# print("✅ Model training completed and saved!")


