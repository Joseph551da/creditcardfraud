# Credit Card Fraud Detection

Welcome to the **Credit Card Fraud Detection** project! This project demonstrates the use of a neural network to detect fraudulent transactions from a credit card dataset.

## Features

- **Data Preprocessing**: The dataset is normalized and shuffled to ensure better performance of the model.
- **Model Training**: A neural network model is trained using Keras with TensorFlow backend.
- **Class Weights Handling**: Class weights are used to handle the imbalance in the dataset.
- **Model Evaluation**: Precision, Recall, and Accuracy metrics are used to evaluate the model's performance.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the model's predictions.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Keras**
- **TensorFlow**
- **Matplotlib**
- **Scikit-learn**

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd credit-card-fraud-detection
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Dataset**:
    - Download the [credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle and place it in the project directory.

### Usage

1. **Data Preprocessing**:
    - The dataset is read and normalized.

    ```python
    dataset = pd.read_csv("creditcard.csv", header=0)
    dataset["Amount"] = (dataset["Amount"] - dataset["Amount"].min()) / (dataset["Amount"].max() - dataset["Amount"].min())
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    ```

2. **Model Training**:
    - The model is defined and trained on the dataset.

    ```python
    model = Sequential()
    model.add(Dense(14, activation="relu", input_shape=(29,)))
    model.add(Dense(7, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=train_x, y=train_y, epochs=10, class_weight=class_weights)
    ```

3. **Model Evaluation**:
    - The model is evaluated on the test set.

    ```python
    score = model.evaluate(x=test_x, y=test_y)
    print("Loss =", score[0])
    print("Precision =", score[1])
    print("Recall =", score[2])
    print("Accuracy =", score[3])
    ```

4. **Plotting Confusion Matrix**:
    - The confusion matrix is plotted to visualize the model's performance.

    ```python
    y_pred = model.predict(x=test_x)
    cnf_matrix = confusion_matrix(test_y, (y_pred > 0.5).astype(int))
    plot_confusion_matrix(cnf_matrix, classes=range(2))
    plt.show()
    ```

## Contributing

We welcome contributions to improve this project. Here's how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add a feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset is provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Thanks to the open-source community for their valuable resources.

---

Thank you for exploring the **Credit Card Fraud Detection** project! If you have any questions or feedback, feel free to open an issue in the repository.
