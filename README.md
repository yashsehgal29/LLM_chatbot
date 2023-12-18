# LLM_chatbot
This LLM chat-bot is A Web App where users can upload their .pdf/.txt/.docx/.doc files having the texts and then they can ask questions from the chatbot related to that text and according to the question, it will answer based on the text provided.

## Steps to be followed before running the code in your environment
1)Fork the repo.<br>
2) Create a virtual environment in the same directory as the files are there by the command "venv name of the virtual environment say env"<br>
3)Then run "env\Scripts\activate" to activate the virtual environment.<br>
4) Then run pip install -r requirements.txt to install all the dependencies and libraries.<br>
5) Create a .env file and make a variable in it with the name "REPLICATE_API_TOKEN=" and provide it your replicate API token.<br>
6)You can run the server using "streamlit run app.py"