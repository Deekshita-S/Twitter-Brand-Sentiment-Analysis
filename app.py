import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer

st.set_page_config(page_title="Brand Sentiment Analyzer", layout="centered")

st.title("🐦 Twitter’s Take on Your Brand")
st.markdown("### Sweet Treat or Bitter Tweet? Paste a tweet below and we'll help you decide.")


model_path = 'bert_model'


tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


st.markdown("#### 📝 Enter Tweet:")
user_input = st.text_input('',placeholder="E.g., I can't believe how bad the service was from @BrandName today.")


submit =st.button('🔍 Analyze Sentiment')

if submit:
    input = tokenizer(user_input, return_tensors='pt',truncation=True,padding=True)
    output = model(**input)
    probability = output.logits.softmax(dim=1)
    label = probability.argmax().item()
    if label==0:
        sentiment = '😐 Plain Post'
    elif label==1:
        sentiment ="🍬 Sweet Treat"
    else:
        sentiment = '🍋 Bitter tweet'
    st.write(f'Sentiment: {sentiment}')
    st.write(f"Confidence: {probability[0][label].item():.2f}")
