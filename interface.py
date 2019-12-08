from main import ChatBot
import streamlit as st
import json
import pickle

bot = ChatBot()

# set this to true to reload data and retrain the model
retrain = False

with open('intents/intents.json') as intent_file:
    data = json.load(intent_file)

if retrain:
    bot.prep_data(data)

with open('data.pickle', 'rb') as f:
    words, labels, train, output = pickle.load(f)

model = bot.init_model(train, output)

if retrain:
    model.fit(train, output, n_epoch=900, batch_size=8, show_metric=True)
    model.save('model.tflearn')
else:
    model.load('model.tflearn')

st.markdown('## Start chatting')
inp = st.text_input('')

if inp != '':
    resp = bot.get_bot_response(inp, data, labels, model, words)

for i in range(len(bot.inputs)):
    st.markdown(f'**Bot**: {bot.responses[i]}')
    st.markdown(f'**You**: {bot.inputs[i]} \n ---')
