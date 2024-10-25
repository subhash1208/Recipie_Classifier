# Recipe Classifier

This project contains a recipe classifier model that can classify recipes into 10 categories.

## Requirements

- Python 3.8 or higher
- The following packages:
  - numpy==1.20.0
  - pandas==1.3.5
  - scikit-learn==1.0.2
  - tensorflow==2.7.0
  - keras==2.7.0
  - matplotlib==3.5.1
  - seaborn==0.11.2
  - nltk==3.6.5
  - transformers==4.12.5

You can install the required packages using the following command:
```bash
   pip install -r Requirements.txt
   ```

## Code

The complete code for the Recipe Classifier project is included in the `Recipe_Classifier.ipynb` Jupyter Notebook.

### Data Preprocessing

The code starts by loading the recipe data from a CSV file using `pandas`. The data is then preprocessed using the following steps:

* **Tokenization**: The recipe text data is tokenized using the `Tokenizer` class from `tensorflow.keras.preprocessing.text`. This involves splitting the text into individual words or tokens.
* **Stopword removal**: Common stopwords such as "the", "and", etc. are removed from the tokenized data using the `stopwords` corpus from `nltk`.
* **Lemmatization**: The tokens are then lemmatized using the `WordNetLemmatizer` class from `nltk`. This involves reducing words to their base or dictionary form.
* **Vectorization**: The preprocessed data is then vectorized using the `pad_sequences` function from `tensorflow.keras.preprocessing.sequence`. This involves converting the tokenized data into numerical vectors of a fixed length.

### Model Architecture

The model architecture consists of the following layers:

* **Input layer**: The input layer takes in the vectorized recipe data with a shape of `(batch_size, sequence_length, embedding_dim)`.
* **Embedding layer**: The input data is then passed through an embedding layer with 128 dimensions.
* **Convolutional layer**: The embedded data is then passed through a convolutional layer with 64 filters, a kernel size of 3, and a ReLU activation function.
* **Max pooling layer**: The output of the convolutional layer is then passed through a max pooling layer with a pool size of 2.
* **Flatten layer**: The output of the max pooling layer is then flattened into a 1D array.
* **Dense layer**: The flattened data is then passed through a dense layer with 128 units and a ReLU activation function.
* **Output layer**: The output of the dense layer is then passed through an output layer with 10 units and a softmax activation function.

### Model Training

The model is trained using the `fit` method from `tensorflow.keras.Model`. The training data is split into training and validation sets using the `train_test_split` function from `sklearn.model_selection`. The model is trained for 10 epochs with a batch size of 32 and a learning rate of 0.001.

### Model Evaluation

The model is evaluated using the `evaluate` method from `tensorflow.keras.Model`. The evaluation metrics include accuracy, precision, recall, and F1-score.

### Visualization

The model's performance metrics are visualized using `matplotlib` and `seaborn`. The accuracy and F1-score are plotted against the number of epochs.

## Usage

To use the code, follow these steps:

1. Download the `recipe_data.csv` dataset and place it in the project directory.
2. Run the `Recipe_Classifier.ipynb` notebook to train and evaluate the model.
3. Use the trained model to classify new recipes.
