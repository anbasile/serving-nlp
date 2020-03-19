import re
import requests
import tensorflow_datasets as tfds
import time

def predict(byte_encoded_texts: bytes) -> str:
    """
        This function calls the TensorRT server,
        sending text encoded as byte and gives back
        a label, either 'positive' or 'negative', depending
        on the sentiment of the text.
    """

    # encode length of string as byte
    string_length = len(byte_encoded_texts).to_bytes(4, 'little')

    # prepend length to actula string
    data = string_length + byte_encoded_texts

    # compute the complete content length
    content_length = len(data)

    response = requests.post(
        url='http://localhost:8000/api/infer/sentiment-model',
        params={'format': 'binary'},
        headers={
            'Content-Type': 'application/octet-stream',
            'Accept': '*/*',
            'Host': 'localhost:8000',
            'NV-InferRequest':
                'batch_size: 1 \
                input { name: "input_1" dims: 1 batch_byte_size: %s } \
                output { name: "dense" cls { count : 1 } }' % content_length},
        data=data)
    if response.ok:
        labels = (re.findall('(positive|negative)', response.text))
    else:
        labels = ['']
    return labels[0]


def run():
    data = tfds.load(
        name='imdb_reviews',
        split='test',
        as_supervised=True)

    texts, labels = zip(*tfds.as_numpy(data))

    assert isinstance(texts[0], bytes)

    labels = [x.item() for x in labels[:10]]

    labels = ['positive' if x == 1 else 'negative' for x in labels]

    start = time.time()

    predictions = [predict(t) for t in texts[:10]]

    end = time.time()

    accuracy = sum([x == y for x, y in zip(predictions[:10], labels[:10])])/10

    print(f'Accuracy: {accuracy}')

    print(f'Elapsed time (seconds): {end-start}')


if __name__ == "__main__":
    run()
