#!/usr/bin/env python
__author__ = "Angelo Basile"
__copyright__ = "Copyright 2020"
__status__ = "Demo"

import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(42)

def _preprocess(text: tf.string)->tf.Tensor:
    """
        This function vectorizes the text, separating
        strings into tokens at every white-space and then
        using a hashing function to map tokens to ids.
    """
    tokens = tf.strings.split(text, ' ')
    ids = tf.strings.to_hash_bucket_fast(tokens, 19999)
    return ids.to_tensor(default_value=20000)

def build_model()->tf.keras.Model:

    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string)
    h = tf.keras.layers.Lambda(_preprocess)(input_layer)
    h = tf.keras.layers.Embedding(20001, 32)(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    output_layer = tf.keras.layers.Dense(2, 'softmax')(h)

    model = tf.keras.Model(
        inputs=input_layer, 
        outputs=output_layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def main():

    data = tfds.load(
        name='imdb_reviews', 
        split='train', 
        as_supervised=True)

    texts, labels = zip(*tfds.as_numpy(data))

    # let's convert byte to strings first
    texts = [x.decode('utf-8') for x in texts]

    labels = [x.item() for x in labels]

    model = build_model()

    model.fit(x=texts, y=labels,epochs=15, batch_size=512)

    # serialize to disk using the SavedModel format
    model.save(
        filepath='./sentiment-model/1/model.savedmodel/',
        save_format='tf')

    loaded_model = tf.keras.models.load_model(
        './sentiment-model/1/model.savedmodel/'
        )

    test_sentences = [
        'I loved the pizza',
        'I hated the pizza']

    # [[4.1961043e-26 1.0000000e+00] [3.0753494e-03 9.9692470e-01]]
    predicted_probabilities = loaded_model.predict(test_sentences)

    # ['positive', 'negative']
    labels = ['positive' if x == 1 else 'negative' for x in tf.argmax(predicted_probabilities)]

    assert labels[0] == 'positive' # I loved the pizza
    assert labels[1] == 'negative' # I hated the pizza

if __name__ == "__main__":
    main()
