


embed_size = 300
max_features = None
maxlen = 50



questions = pad_sequences(questions, maxlen=maxlen)
answers = pad_sequences(answers, maxlen=maxlen)

word_index = tokenizer.word_index
max_features = len(word_index) + 1

print(word_index)

embedding_matrix_1 = load_glove(word_index, max_features)
embedding_matrix_2 = load_fasttext(word_index, max_features)
embedding_matrix = np.mean((embedding_matrix_1, embedding_matrix_2), axis=0)
del embedding_matrix_1, embedding_matrix_2
gc.collect()

encoder_input_data = np.zeros((len(questions), 14),dtype='float32')
decoder_input_data = np.zeros((len(answers), 50),dtype='float32')
decoder_target_data = np.zeros((len(answers), 50, max_features),dtype='float32')

for i, (input_text, target_text) in enumerate(zip(dataset['Questions'].values, dataset['Answers'].values)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = word_index[word]
    for t, word in enumerate(target_text.split()):
        decoder_input_data[i, t] = word_index[word]
        if t > 0:
            decoder_target_data[i, t - 1, word_index[word]] = 1

#Model for encoder - decoder
encoder_inputs = Input(shape=(None,))
en_x = Embedding(264, embed_size, weights = [embedding_matrix])(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dex = Embedding(264, embed_size)
final_dex = dex(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)
decoder_dense = Dense(264, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
callback = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0, mode='auto')]
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=1, epochs=200, validation_split=0.05,
          callbacks= callback)
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)

model.save('chatbot.h5')

"""
Error here solve below this - till here model is saved
"""
#Sampling
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in word_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in word_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 264))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word_index["_start_"]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == "_end_" or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 264))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', dataset['Questions'].values[seq_index])
    print('Decoded sentence:', decoded_sentence)