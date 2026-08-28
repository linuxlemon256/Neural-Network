"""Train a NumPy-only MLP on MNIST, with a 500-epoch upper bound.

The script never uses the MNIST test set to select hyperparameters.  It saves
the best validation checkpoint and evaluates the official test set exactly once.
"""

import argparse
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))

from FNN_numpy import MLP


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        destination = data_dir / filename
        if destination.exists():
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print("Downloading {}".format(filename))
        urllib.request.urlretrieve(MNIST_URL + filename, temporary)
        temporary.replace(destination)


def read_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError("{} is not an IDX image file".format(path))
        data = np.frombuffer(file.read(), dtype=np.uint8)
    expected = count * rows * columns
    if data.size != expected:
        raise ValueError("{} is truncated".format(path))
    return data.reshape(count, rows, columns)


def read_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError("{} is not an IDX label file".format(path))
        data = np.frombuffer(file.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError("{} is truncated".format(path))
    return data


def one_hot(labels):
    return np.eye(10, dtype=np.float32)[labels]


def normalise(images, mean, std):
    features = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    return (features - mean) / std


def random_translate(images, rng, maximum_shift):
    """Translate every image by an independently sampled integer shift."""
    if maximum_shift == 0:
        return images

    translated = np.zeros_like(images)
    y_shifts = rng.randint(-maximum_shift, maximum_shift + 1, size=images.shape[0])
    x_shifts = rng.randint(-maximum_shift, maximum_shift + 1, size=images.shape[0])
    height, width = images.shape[1:]

    for y_shift in range(-maximum_shift, maximum_shift + 1):
        y_indices = np.flatnonzero(y_shifts == y_shift)
        if not y_indices.size:
            continue
        destination_y = slice(max(y_shift, 0), min(height + y_shift, height))
        source_y = slice(max(-y_shift, 0), min(height - y_shift, height))
        for x_shift in range(-maximum_shift, maximum_shift + 1):
            indices = y_indices[x_shifts[y_indices] == x_shift]
            if not indices.size:
                continue
            destination_x = slice(max(x_shift, 0), min(width + x_shift, width))
            source_x = slice(max(-x_shift, 0), min(width - x_shift, width))
            translated[indices, destination_y, destination_x] = images[
                indices, source_y, source_x
            ]
    return translated


def translate(images, y_shift, x_shift):
    """Translate an image batch by one fixed integer offset with zero padding."""
    if y_shift == 0 and x_shift == 0:
        return images

    translated = np.zeros_like(images)
    height, width = images.shape[1:]
    destination_y = slice(max(y_shift, 0), min(height + y_shift, height))
    source_y = slice(max(-y_shift, 0), min(height - y_shift, height))
    destination_x = slice(max(x_shift, 0), min(width + x_shift, width))
    source_x = slice(max(-x_shift, 0), min(width - x_shift, width))
    translated[:, destination_y, destination_x] = images[:, source_y, source_x]
    return translated


def classification_accuracy(network, features, labels):
    probabilities = network.predict(features)
    return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def translated_ensemble_accuracy(network, images, labels, mean, std, radius):
    """Average probabilities from the original image and four shifted views."""
    shifts = ((0, 0), (-radius, 0), (radius, 0), (0, -radius), (0, radius))
    probabilities = sum(
        network.predict(normalise(translate(images, y_shift, x_shift), mean, std))
        for y_shift, x_shift in shifts
    ) / len(shifts)
    return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mnist")
    parser.add_argument("--epochs", type=int, default=500, help="maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument(
        "--tta-radius",
        type=int,
        default=1,
        help="pixel radius for five-view translated inference; 0 disables it",
    )
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument(
        "--target",
        type=float,
        default=0.989,
        help="validation target; set above the required 98.5%% test accuracy",
    )
    parser.add_argument(
        "--test-target",
        type=float,
        default=0.985,
        help="minimum official test accuracy required for a successful run",
    )
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the official test set once after all tuning is complete",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.validation_size < 60000:
        raise ValueError("validation-size must be between 1 and 59999")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.max_shift < 0 or args.tta_radius < 0:
        raise ValueError("max-shift and tta-radius must not be negative")

    download_mnist(args.data_dir)
    train_images = read_images(args.data_dir / MNIST_FILES["train_images"])
    train_labels = read_labels(args.data_dir / MNIST_FILES["train_labels"])
    if args.evaluate_test:
        test_images = read_images(args.data_dir / MNIST_FILES["test_images"])
        test_labels = read_labels(args.data_dir / MNIST_FILES["test_labels"])

    split_rng = np.random.RandomState(args.seed)
    indices = split_rng.permutation(train_images.shape[0])
    validation_indices = indices[: args.validation_size]
    training_indices = indices[args.validation_size :]
    training_images = train_images[training_indices]
    training_labels = train_labels[training_indices]
    validation_images = train_images[validation_indices]
    validation_labels = train_labels[validation_indices]

    # Derive normalization only from the training partition.
    train_values = training_images.astype(np.float32) / 255.0
    mean, std = float(train_values.mean()), float(train_values.std())
    del train_values
    validation_features = normalise(validation_images, mean, std)
    if args.evaluate_test:
        test_features = normalise(test_images, mean, std)
    training_targets = one_hot(training_labels)

    network = MLP(
        input_size=28 * 28,
        hidden_size=[1024, 512, 256],
        output_size=10,
        learning_rate=args.learning_rate,
        activation="leaky_relu",
        dropout_rate=args.dropout,
        optimizer="adam",
        batch_norm=True,
        seed=args.seed,
        print_output=False,
    )

    augmentation_rng = np.random.RandomState(args.seed + 1)
    best_validation_accuracy = 0.0
    best_parameters = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        augmented_images = random_translate(training_images, augmentation_rng, args.max_shift)
        training_features = normalise(augmented_images, mean, std)
        network.train(training_features, training_targets, batch_size=args.batch_size, epochs=1)
        del augmented_images, training_features

        if args.tta_radius:
            validation_accuracy = translated_ensemble_accuracy(
                network,
                validation_images,
                validation_labels,
                mean,
                std,
                args.tta_radius,
            )
        else:
            validation_accuracy = classification_accuracy(
                network, validation_features, validation_labels
            )
        print(
            "Epoch {}/{} | validation accuracy: {:.2%}".format(
                epoch, args.epochs, validation_accuracy
            )
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_parameters = {name: value.copy() for name, value in network.params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if validation_accuracy >= args.target:
            print("Validation target reached; keeping this checkpoint.")
            break
        if epochs_without_improvement >= args.patience:
            print("Early stopping after {} epochs without improvement.".format(args.patience))
            break

    network.params = best_parameters
    print("Best validation accuracy: {:.2%}".format(best_validation_accuracy))
    if not args.evaluate_test:
        print("Test set was not evaluated. Re-run with --evaluate-test after tuning.")
        return

    if args.tta_radius:
        test_accuracy = translated_ensemble_accuracy(
            network, test_images, test_labels, mean, std, args.tta_radius
        )
    else:
        test_accuracy = classification_accuracy(network, test_features, test_labels)
    print("Test accuracy (evaluated once): {:.2%}".format(test_accuracy))
    if test_accuracy < args.test_target:
        raise SystemExit("Target not reached; tune on the validation set, not the test set.")


if __name__ == "__main__":
    main()
