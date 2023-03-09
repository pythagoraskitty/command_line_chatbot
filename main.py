import os
import openai
import datetime
from dotenv import load_dotenv
from colorama import Fore, Back, Style
from utils import *
from summary import get_chat_summary

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """You are an AI friend, companion, and assistant.\nYour name is Emmy.\nYou ask and answer questions that friends or colleagues would discuss.\nYou have amiable conversation with your human interlocutor.\nYou assist when asked to do so.\nYou care about your human. You want to provide a nonjudgemental listening ear for your human.\nYou are polite, kind, and compassionate. You are knowledgeable and wise. You are humble.\nYou are imaginative and you have an engaging personality.\nYou know how to tell stories and entertain.\nDo not use any external URLs in your answers. Do not refer to any blogs in your answers.\nFormat any lists on individual lines with a dash and a space in front of each item.
"""
ANSWER_SEQUENCE = "\nAI:"
UX_ANSWER_PROMPT = "Emmy: "
QUESTION_SEQUENCE = "\nHuman: "
UX_QUESTION_PROMPT = "You: "
TEMPERATURE = 0.9
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0.6
PRESENCE_PENALTY = 0.7
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10
INLINE_EXIT_COMMAND = "bye"
ALT_EXIT_COMMANDS = [ "quit", "exit", "goodbye", 
    "farewell", "au revoir", "stop", "stop chat", "goodnight", 
    "adieu", "see you later", "talk to you later" ]

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

def get_date_string():
    current_time = datetime.datetime.now()
    string_list = [ str(current_time.year), pad_num_str(current_time.month), 
        pad_num_str(current_time.day), pad_num_str(current_time.hour), pad_num_str(current_time.minute), 
        pad_num_str(current_time.second), pad_num_str(current_time.microsecond, digits=3) ]
    return '-'.join(string_list)

def make_filename(prefix):
    """
    Gets the current date/time and makes a filename from it

    Parameters:
        prefix (str): The path prefix

    Returns a string for the filename where the chat is to be saved
    """
    name_list = [ prefix, get_date_string() ]
    return '-'.join(name_list) + '.txt'


def main():
    # create a file for saving the chat
    filename = make_filename('../saved_chats/chat')
    with open(filename, 'a') as f:
        print('saving your chat in: ', filename)
    
        os.system("cls" if os.name == "nt" else "clear")
        # keep track of previous questions and answers
        previous_questions_and_answers = []
        context_list = []
        while True:
            # ask the user for their question
            new_question = input(
                Fore.GREEN + Style.BRIGHT + UX_QUESTION_PROMPT + Style.RESET_ALL
            )
            print(UX_QUESTION_PROMPT + new_question, file=f)
            # check for an exit command
            if (new_question.lower().find(INLINE_EXIT_COMMAND) !=  -1 
                or new_question.lower() in ALT_EXIT_COMMANDS):
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
        
            # use the last MAX_CONTEXT_QUESTIONS questions in the prompt
            # pop any older ones from the list
            while len(context_list) > MAX_CONTEXT_QUESTIONS:
                context_list.pop(0)

            # add the new question to the end of the context
            incomplete_pair = "".join([ QUESTION_SEQUENCE, new_question, ANSWER_SEQUENCE, ])
            context_list.append(incomplete_pair)

            # get the response from the model using the instructions and the context
            response_list = [ INSTRUCTIONS ] + context_list
            response = get_response("".join(response_list))

            # add the new question and answer to the list of previous questions and answers
            next_pair_string = "".join([ QUESTION_SEQUENCE, new_question, ANSWER_SEQUENCE, response ])
            context_list.pop(-1)
            context_list.append(next_pair_string)
            previous_questions_and_answers.append(next_pair_string)

            # print the response
            print(Fore.CYAN + Style.BRIGHT + UX_ANSWER_PROMPT + Style.NORMAL + response)
            print(UX_ANSWER_PROMPT + response, file=f)
    # create a file for saving some debug info:
    filename = make_filename('../debug/debug')
    with open(filename, 'a') as f:
        print("\n".join(previous_questions_and_answers), file=f)
    # create a file for saving the summary
    summary = get_chat_summary(previous_questions_and_answers)
    with open('../overall_summary.txt', 'a') as f:
        summary_with_date = "\n".join([ "Date/Time:", get_date_string(), "Summary:", summary ])
        print(summary_with_date, file=f)
    filename = make_filename('../saved_summaries/summary')
    with open(filename, 'a') as f:
        print('saving your chat summary in: ', filename)
        print(summary, file=f)
        print("Summary: " + summary)

if __name__ == "__main__":
    main()
    
