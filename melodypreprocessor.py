

import json

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer


class MelodyPreprocessor:


    def __init__(self, dataset_path, batch_size=32):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        Returns:
            tf_training_dataset: A TensorFlow dataset containing input-target
                pairs suitable for training a sequence-to-sequence model.
        """
        dataset = self._load_dataset()
        self._set_max_melody_length(dataset)
        self.number_of_tokens = 12         # max number of notes
        input_sequences, target_sequences = self._create_sequence_pairs(dataset)
        tf_training_dataset = self._convert_to_tf_dataset(
            input_sequences, target_sequences
        )
        return tf_training_dataset

    def _load_dataset(self):
        """
        Loads the melody dataset from a JSON file.

        Returns:
            list: A list of melodies from the dataset.
        """
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum melody length based on the dataset.

        Parameters:
            melodies (list): A list of tokenized melodies.
        """
        
        self.max_melody_length = max(len(melody[0]) for melody in melodies)


    def _create_sequence_pairs(self, melodies):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies (list): A list of tokenized melodies.

        Returns:
            tuple: Two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, (len(melody[0])+1)):
                input_seq = [melody[0][:i], melody[1][:i], melody[2][:i]]
                target_seq = [melody[0][1 : i + 1], melody[1][1 : i + 1], melody[2][1 : i + 1]]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)


        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list): The sequence to be padded.

        Returns:
            list: The padded sequence.
        """
        for i in sequence:
            return [sequence[0] + [0] * (self.max_melody_length - len(sequence[0])), sequence[1] + [0] * (self.max_melody_length - len(sequence[0])), sequence[2] + [0] * (self.max_melody_length - len(sequence[0]))]
        

    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a TensorFlow Dataset.

        Parameters:
            input_sequences (list): Input sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled
                TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset


if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("dataset-integers-3voices.json", batch_size=32)
    training_dataset = preprocessor.create_training_dataset()
