from main import ChatBot
import streamlit as st
import json
import pickle

bot = ChatBot()

retrain = False

inputs = []
responses = []

with open('intents/intents.json') as intent_file:
    data = json.load(intent_file)

try:
    x
    with open('data.pickle', 'rb') as f:
        words, labels, train, output = pickle.load(f)
except:
    bot.prep_data(data)
    with open('data.pickle', 'rb') as f:
        words, labels, train, output = pickle.load(f)

model = bot.init_model(train, output)

if retrain:
    model.fit(train, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')
else:
    model.load('model.tflearn')

inp = st.text_input('Start chatting')
# inputs.insert(0, inp)

resp = bot.get_bot_response(inp, data, labels, model, words)
# responses.insert(0, inp)

if inp != '':
    st.markdown(f'### {resp}')

    # not working because arrays get cleared each time i submit
    # for i in range(len(inputs)):
    #     st.write(f'You: {inputs[i]}')
    #     st.write(f'Bot: {responses[i]}')
