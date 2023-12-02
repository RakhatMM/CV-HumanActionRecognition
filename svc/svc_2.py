import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import itertools
from joblib import dump

label_order = ['Diving-Side', 'Golf-Swing', 'Kicking', 'Lifting', 'Riding-Horse', 'SkateBoarding-Front',
               'Swing-Bench', 'Swing-SideAngle', 'boxing', 'handclapping', 'handwaving',
               'jogging', 'running', 'walking']

def load_vlad_data_and_labels(root_folder):
    """Load VLAD data and corresponding labels."""
    data, labels, class_names = [], [], []
    for action_class in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, action_class)

        # Skip if not a directory
        if not os.path.isdir(class_folder):
            continue

        class_names.append(action_class)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.npy'):
                vlad_vector = np.load(os.path.join(class_folder, file_name))
                data.append(vlad_vector)
                labels.append(len(class_names) - 1)  # Assign based on current class_names length

    return np.array(data), np.array(labels), class_names

def calculate_top_k_accuracy_per_class(y_true, y_scores, k, num_classes):
    """Calculate top-k accuracy for each class."""
    top_k_predictions = np.argsort(y_scores, axis=1)[:, -k:]
    accuracies = np.zeros(num_classes)

    for i in range(num_classes):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) == 0:
            continue  # Skip classes not present in the test set

        class_true = y_true[class_indices]
        class_scores = y_scores[class_indices]
        matches = np.any(top_k_predictions[class_indices] == class_true[:, None], axis=1)
        accuracies[i] = np.mean(matches)

    return accuracies

# previous code for plot_confusion
# def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
#     """This function prints, plots, and saves the confusion matrix."""
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     # Save the figure before showing
#     plt.savefig(save_path, format='png', bbox_inches='tight')
#
#     plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, label_order, title='Confusion matrix', cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    """
    This function prints, plots, and saves the confusion matrix with labels ordered as specified.
    """
    # Create a mapping from the class names to indices in the desired label order
    label_mapping = {class_name: label_order.index(class_name) for class_name in class_names}

    # Map the true and predicted labels to the corresponding indices in the label order
    y_true_mapped = np.array([label_mapping[class_names[label]] for label in y_true])
    y_pred_mapped = np.array([label_mapping[class_names[label]] for label in y_pred])

    # Compute confusion matrix using the indices in the label order
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=range(len(label_order)))

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set the ticks to be the class names in the specified order
    tick_marks = np.arange(len(label_order))
    plt.xticks(tick_marks, label_order, rotation=90)
    plt.yticks(tick_marks, label_order)

    # Labeling the plot with correct, incorrect counts, and thresholds for color
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure before showing
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()


# Your function definitions for load_vlad_data_and_labels and plot_confusion_matrix remain the same.

def calculate_top_k_accuracy(y_true, y_scores, k):
    """Calculate the overall top-k accuracy."""
    # Get indices of the top k scores for each sample
    top_k_predictions = np.argsort(y_scores, axis=1)[:, -k:]

    # Initialize a counter for correct top-k predictions
    correct_top_k = 0

    # Iterate over each sample and check if true label is in top k predictions
    for i in range(y_scores.shape[0]):
        if y_true[i] in top_k_predictions[i]:
            correct_top_k += 1

    # Calculate the top-k accuracy
    top_k_accuracy = correct_top_k / y_scores.shape[0]
    return top_k_accuracy


def main():
    root_folder = "vlad_representation"

    # Load VLAD data and corresponding labels
    X, y, class_names = load_vlad_data_and_labels(root_folder)

    # Initialize classifier
    classifier = SVC(kernel='linear', probability=True)

    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Save the trained model
    model_filename = 'trained_svc_model.joblib'
    dump(classifier, model_filename)
    print(f"Model saved as {model_filename}")

    # Predict scores on the test set
    y_scores = classifier.decision_function(X_test)

    # Calculate top-1, top-3, and top-5 accuracies overall
    top1_accuracy = calculate_top_k_accuracy(y_test.reshape(-1, 1), y_scores, 1)
    top3_accuracy = calculate_top_k_accuracy(y_test.reshape(-1, 1), y_scores, 3)
    top5_accuracy = calculate_top_k_accuracy(y_test.reshape(-1, 1), y_scores, 5)

    print(f"Overall Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Overall Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
    print(f"Overall Top-5 Accuracy: {top5_accuracy * 100:.2f}%")

    # Calculate and plot confusion matrix
    y_pred = np.argmax(y_scores, axis=1)
    # prev confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")
    print(class_names)
    plot_confusion_matrix(y_test, y_pred, class_names, label_order)

    # Optionally, calculate and print top-k accuracies for each class
    # ...
    num_classes = len(class_names)
    top1_accuracies = calculate_top_k_accuracy_per_class(y_test, y_scores, 1, num_classes)
    top3_accuracies = calculate_top_k_accuracy_per_class(y_test, y_scores, 3, num_classes)
    top5_accuracies = calculate_top_k_accuracy_per_class(y_test, y_scores, 5, num_classes)

    for i, class_name in enumerate(class_names):
        print(f"Class '{class_name}' - Top-1 Accuracy: {top1_accuracies[i] * 100:.2f}%")
        print(f"Class '{class_name}' - Top-2 Accuracy: {top3_accuracies[i] * 100:.2f}%")
        print(f"Class '{class_name}' - Top-3 Accuracy: {top5_accuracies[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
