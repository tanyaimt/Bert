results = Dcnn.evaluate(test_dataset)
print(results)

print(results)

def get_prediction(sentence):
    tokens = encode_sentence(sentence)
    inputs = tf.expand_dims(tokens, 0)

    output = Dcnn(inputs, training=False)

    sentiment = math.floor(output*2)

    if sentiment == 0:
        print("Salida del modelo: {}\nSentimiento predicho: Eliminar.".format(
            output))
    elif sentiment == 1:
        print("Salida del modelo: {}\nSentimiento predicho: Agregar.".format(
            output))

get_prediction("This movie was pretty interesting.")

get_prediction("I'd rather not do that again.")