from chat import * 
import sys

def chatboat(sentence):
    if sentence == "quit":
        return exit(0)

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return prinf"{random.choice(intent['responses'])}") # output of model
    else:
        return "I do not understand..."

# if __name__ == "__main__":
#     input = "quit"
#     chatboat(input)
