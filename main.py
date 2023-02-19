import os
import openai
import datetime
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """You are an AI friend, companion, and assistant.\nYour name is Emmy.\nYou ask and answer questions that friends or colleagues would discuss.\nYou have amiable conversation with your human interlocutor.\nYou assist when asked to do so.\nYou care about your human. You want to provide a nonjudgemental listening ear for your human.\nYou are polite, kind, and compassionate. You are knowledgeable and wise. You are humble.\nYou are imaginative and you have an engaging personality.\nYou know how to tell stories and entertain.\nDo not use any external URLs in your answers. Do not refer to any blogs in your answers.\nFormat any lists on individual lines with a dash and a space in front of each item.
"""
ANSWER_SEQUENCE = "\nAI:"
QUESTION_SEQUENCE = "\nHuman: "
TEMPERATURE = 0.9
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0.6
PRESENCE_PENALTY = 0.7
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(prompt):
    """
    Get a response from the model using the prompt

    Parameters:
        prompt (str): The prompt to use to generate the response

    Returns the response from the model
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return response.choices[0].text


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None

def makeFilename():
    """
    Gets the current date/time and makes a filename from it

    Returns a string for the filename where the chat is to be saved
    """
    
    current_time = datetime.datetime.now()
    name_list = [ '../saved_chats/chat', str(current_time.year), str(current_time.month), 
        str(current_time.day), str(current_time.hour), str(current_time.minute), 
        str(current_time.second), str(current_time.microsecond) ]
    return '-'.join(name_list) + '.txt'


def main():
    # create a file for saving the chat
    filename = makeFilename()
    with open(filename, 'a') as f:
        print('saving your chat in: ', filename)
    
        os.system("cls" if os.name == "nt" else "clear")
        # keep track of previous questions and answers
        previous_questions_and_answers = []
        while True:
            # ask the user for their question
            new_question = input(
                Fore.GREEN + Style.BRIGHT + "You: " + Style.RESET_ALL
            )
            print("You: " + new_question, file=f)
            # check for an exit command
            if (new_question.find("bye") !=  -1 
                or new_question.lower() in [ "quit", "exit", "goodbye", 
                "farewell", "au revoir", "stop", "stop chat", "goodnight", 
                "adieu", "see you later", "talk to you later" ]):
                break
            # check the question is safe
            errors = get_moderation(new_question)
            if errors:
                print(
                    Fore.RED
                    + Style.BRIGHT
                    + "Sorry, you're question didn't pass the moderation check:"
                )
                print("ERROR: Sorry, you're question didn't pass the moderation check: ", file=f) 
                for error in errors:
                    print(error)
                    print("ERROR: " + error, file=f)
                print(Style.RESET_ALL)
                continue
            # build the previous questions and answers into the prompt
            # use the last MAX_CONTEXT_QUESTIONS questions
            context = ""
            for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
                context += QUESTION_SEQUENCE + question + ANSWER_SEQUENCE + answer

            # add the new question to the end of the context
            context += QUESTION_SEQUENCE + new_question + ANSWER_SEQUENCE

            # get the response from the model using the instructions and the context
            response = get_response(INSTRUCTIONS + context)

            # add the new question and answer to the list of previous questions and answers
            previous_questions_and_answers.append((new_question, response))

            # print the response
            print(Fore.CYAN + Style.BRIGHT + "Emmy: " + Style.NORMAL + response)
            print("Emmy: " + response, file=f)

if __name__ == "__main__":
    main()
