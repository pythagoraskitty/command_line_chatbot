import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(string))

def pad_num_str(num, digits = 2):
    if num < 0:
        raise ValueError("invalid number for string conversion, must be nonnegative")
    for d in range(1, digits):
        power = pow(10, d)
        if num < power:
            return (digits - d) * '0' + str(num)
    return str(num)
