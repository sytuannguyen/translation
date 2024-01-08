import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

def main():
	st.Title('French to English tranlation, by Dr Tuan Nguyen-Sy')

	input_text = st.text_input('Enter a sentense in French here','La vie est belle')

	st.button('Translate'):
		st.text('Translating ...')
		
		model_name = 'Helsinki-NLP/opus-mt-fr-en'
		tokenizer = MarianTokenizer.from_pretrained(model_name)
		model = MarianMTModel.from_pretrained(model_name)

		input_token_numbers = tokenizer.encode(input_text,output_tensors='pt')
		output_token_numbers = model.generate(input_token_numbers)

		output_text = tokenizer.decode(output_token_numbers[0], skip_special_tokens=True)

		st.success(output_text)
	
if __name__ == __main__():
	main()
