
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
from midiutil import MIDIFile

# Global parameters
EPOCHS = 100
BATCH_SIZE = 32
DATA_PATH = "dataset-integers-3voices.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()


def train(train_dataset, transformer, epochs):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for (batch, (input, target)) in enumerate(train_dataset):
            # Perform a single training step
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            print(
                f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}"
            )


@tf.function
def _train_step(input, target, transformer):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input (tf.Tensor): The input sequences.
        target (tf.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.

    Returns:
        tf.Tensor: The loss value for the training step.
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position
    target_input = _right_pad_sequence_once(target[:, :, :-1])
    target_real = _right_pad_sequence_once(target[:, :, 1:])

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        # TODO: Add padding mask for encoder + decoder and look-ahead mask
        # for decoder
        predictions = transformer(input, target_input, True, None, None, None)

        # Compute loss between the real output and the predictions
        loss = _calculate_loss(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's parameters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (tf.Tensor): The actual target sequences.
        pred (tf.Tensor): The predicted sequences by the model.

    Returns:
        average_loss (tf.Tensor): The computed loss value.
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (tf.Tensor): The sequence to be padded.

    Returns:
        tf.Tensor: The padded sequence.
    """
    return tf.pad(sequence, [[0, 0], [0, 0], [0, 1]], "CONSTANT")

def create_midi_from_string(notes, output_file='output.mid'):
    # Create a MIDI file
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0

    # Set track name and tempo
    midi.addTrackName(track, time, "Generated MIDI")
    midi.addTempo(track, time, 480)  # Adjust tempo as needed

    # Map integers to MIDI note numbers
    note_mapping = {
        1: 60,  # C
        2: 62,  # D
        3: 64,  # E
        4: 65,  # F
        5: 67,  # G
        6: 69,  # A
        7: 71,  # B
        8: 72,  # High C
        9: 74,
        10: 76,
        11: 77,
        12: 79
        # Add more mappings if needed
    }

    # Convert string of integers to successive quarter notes
    for triad in notes:
        note1, note2, note3 = triad

        
        midi.addNote(track, 0, note_mapping.get(note1, 60), time, 1, 100)
        midi.addNote(track, 0, note_mapping.get(note2, 60), time, 1, 100)
        midi.addNote(track, 0, note_mapping.get(note3, 60), time, 1, 100)
        time += 1

    # Write the MIDI file
    with open(output_file, "wb") as midi_file:
        midi.writeFile(midi_file)

if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        pe_input_rows=3,
        pe_input_cols=1000,
        pe_target_rows=3,
        pe_target_cols=1000,
        dropout_rate=0.1,
    )

    train(train_dataset, transformer_model, EPOCHS)

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model
    )
    start_sequence = [[[8, 7, 10, 9, 8],
                       [3, 5, 6, 7, 6],
                       [1, 3, 1, 2, 4]]]
    new_melody = melody_generator.generate(start_sequence)
    # print(f"Generated melody: \n{new_melody}")
    input_notes = new_melody
    notes = [(input_notes[0][0][i], input_notes[0][1][i], input_notes[0][2][i]) for i in range(0, len(input_notes[0][0]))]
    create_midi_from_string(notes, output_file='output.mid')
