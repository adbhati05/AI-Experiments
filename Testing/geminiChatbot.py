# Importing dotenv and os so I can access the variables in .env.
import dotenv
import os

# From here, set up your google AI account to be able to import googlegenerativeai
import google.generativeai as genai

# Setting up the generative Gemini model. This model is going to be a chat bot. 
dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
chatBot = model.start_chat()

# Creating a while-loop that terminates when the user inputs "exit" or "Exit" when speaking to the chatbot. 
while True:
    userInput = input("You: ")
    if userInput.lower() == "exit":
        print("Chatbot: Have a goontastic day!")
        break
    response = chatBot.send_message(userInput)
    print("Chatbot: ", response.text)