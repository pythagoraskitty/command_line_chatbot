import platform
import os
import openai
from functools import reduce
from dotenv import load_dotenv
from utils import *

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

SUMMARY_INSTRUCTIONS = "Summarize this chat transcript: \n"
CONSOLIDATE_INSTRUCTIONS = "Consolidate these chat summaries: \n"

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
MAX_INPUT = 4000 - MAX_TOKENS
MAX_CONSOLIDATED = 1500
MAX_OVERLAP = 200

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

def get_token_counts(string_list):
    return list(map(num_tokens_from_string, string_list))

def get_break_point(string):
    length = len(string)
    endline = string.find('\n', 1)
    if endline != -1 and endline <= length // 2:
        return endline
    period = string.find('.', 1)
    if period == -1:
        period += length
    question = string.find('?', 1)
    if question == -1:
        question += length
    exclamation = string.find('!', 1)
    if exclamation == -1:
        exclamation += length
    break_point = min([ period, question, exclamation ])
    if break_point < length - 1:
        return break_point
    space = string.rfind(' ', 1, length // 2)
    if space > 1:
        return space
    return length // 2

def make_chunks(string_list, token_counts):
    chunks = []
    i = 0
    num_overlap_tokens = 0
    overlap = ""
    while i < len(string_list):
        prev_index = i
        next_chunk_list = [overlap]
        next_token_total = num_overlap_tokens
        while i < len(string_list) and next_token_total + token_counts[i] < MAX_INPUT:
            next_chunk_list.append(string_list[i])
            next_token_total += token_counts[i]
            i += 1
        chunks.append("\n".join(next_chunk_list))
        if i < len(string_list) and i > 0:
            overlap = string_list[i - 1]
            num_overlap_tokens = num_tokens_from_string(overlap)
            while num_overlap_tokens > MAX_OVERLAP:
                break_point = get_break_point(overlap)
                overlap = overlap[break_point + 1:]
                num_overlap_tokens = num_tokens_from_string(overlap)
    return chunks

def make_chunk_summary(chunk):
    return get_response(SUMMARY_INSTRUCTIONS + chunk)

def consolidate_summaries_chunk(chunk):
    return get_response(CONSOLIDATE_INSTRUCTIONS + chunk)

def get_chat_summary(questions_and_answers):
    """
    Get a summary of a chat from the model using the prompt

    Parameters:
        questions_and_answers (list of str): A list of strings
        where each string is a question and answer pair

    Returns the response from the model
    """
    qa_token_counts = get_token_counts(questions_and_answers)
    qa_chunks = make_chunks(questions_and_answers, qa_token_counts)
    summaries = list(map(make_chunk_summary, qa_chunks))
    summary_token_counts = get_token_counts(summaries)
    summary_chunks = make_chunks(summaries, summary_token_counts)
    consolidated_chunks = list(map(consolidate_summaries_chunk, summary_chunks))
    consolidated_token_counts = get_token_counts(consolidated_chunks)
    total_count = reduce((lambda x, y: x + y), consolidated_token_counts)
    while total_count > MAX_CONSOLIDATED or len(consolidated_chunks) > 1:
        next_chunks = make_chunks(consolidated_chunks, consolidated_token_counts)
        consolidated_chunks = list(map(consolidate_summaries_chunk, next_chunks))
        consolidated_token_counts = get_token_counts(consolidated_chunks)
        total_count = reduce((lambda x, y: x + y), consolidated_token_counts)
    return consolidated_chunks[0]
