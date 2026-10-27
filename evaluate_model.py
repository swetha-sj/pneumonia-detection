
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the trained model
model = tf.keras.models.load_model("pneumonia_cnn.keras")

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "archive/chest_xray/test",
    image_size=(150, 150),
    batch_size=32,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    shuffle=False
)

# Get actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_data
]).astype(int)

# Get model predictions
predictions = model.predict(test_data)
y_pred = (predictions.ravel() >= 0.5).astype(int)

# Print evaluation report
print("\nCLASSIFICATION REPORT\n")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Pneumonia Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
