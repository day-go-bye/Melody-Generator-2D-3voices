

import tensorflow as tf
import numpy as np


class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, max_length=500):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor)

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_notes = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_notes
            )
            

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = start_sequence
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """

        latest_predictions = predictions[:, :, -1, :]
        print("latest_predictions:")
        print(latest_predictions)

        def apply_temperature(logits, temperature):
            # Ensure that temperature is greater than 0
            temperature = max(1e-8, temperature)

            # Apply temperature scaling to the logits
            logits /= temperature

            # Softmax to get probabilities
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

            return tf.convert_to_tensor(probabilities)
        
        temperature = .5

        # Apply temperature to the logits
        probabilities = apply_temperature(latest_predictions, temperature)
        print("probabilities:")
        print(probabilities)

        predicted_note_indeces = tf.argmax(probabilities[[0]], axis=1)
        print("predicted_note_indeces:")
        print(predicted_note_indeces)

        predicted_note1 = predicted_note_indeces.numpy()[0]
        print("predicted note 1:")
        print(predicted_note1)

        predicted_note2 = predicted_note_indeces.numpy()[1]
        print("predicted note 2:")
        print(predicted_note2)

        predicted_note3 = predicted_note_indeces.numpy()[2]
        print("predicted note 3:")
        print(predicted_note3)

        predicted_notes = [[predicted_note1], [predicted_note2], [predicted_note3]]

        return predicted_notes
    

    def _append_predicted_note(self, input_tensor, predicted_notes):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [predicted_notes]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_melody = generated_sequence.numpy()

        return generated_melody
