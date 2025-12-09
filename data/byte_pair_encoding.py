class BytePairTokenizer:
    def __init__(self, text):
        # create mapping from char -> byte and byte -> char
        self.byte_to_char = get_byte_to_char_mapping()
        self.char_to_byte = {v: k for k, v in self.byte_to_char.items()}

        # create token -> id dictionary
        self.token_ids = {}
        self.current_id = 0
        for char in self.byte_to_char.values():
            self.token_ids[self.current_id] = char
            self.current_id += 1

        # convert text to mapped characters
        text_bytes = text.encode('utf-8')
        
        self.current_tokens = [self.byte_to_char[b] for b in text_bytes]

        # initialize state
        self.merges = []
    
    def train(self, num_merges):
        """
        Runs the training loop until desired vocabulary size is reached
        """

        # set up training loop
        for iw in range(num_merges):
            # get the pairwise stats
            stats = self.get_stats(self.current_tokens)

            # if no more pairs to merge
            if not stats:
                break

            # get most common pair
            best_pair = max(stats, key=stats.get)
            new_token = best_pair[0] + best_pair[1]

            # add it to vocabulary list
            self.merges.append(best_pair)

            # add to token ids dictionary
            self.token_ids[new_token] = self.current_id
            self.current_id += 1

            # merge the tokens
            self.current_tokens = self.merge_tokens(self.current_tokens, best_pair)

    def encode(self, text):
        """
        Encodes the compressed tokens into their corresponding token ids
        """
        text_bytes = text.encode('utf-8')
        curr_tokens = [self.byte_to_char[n] for n in text_bytes]

        # set up loop
        i = 0

        # run loop
        for merge in self.merges:
            curr_tokens = self.merge_tokens(curr_tokens, merge)

        encoded = []
        # loop through compressed list of tokens
        for token in curr_tokens:
            encoded.append(self.token_ids.get(token))
        
        return encoded

    def merge_tokens(self, tokens, pair):
        """
        Takes in the list of tokens and merges the most common token pair, returning a new list of tokens
        
        :param tokens (list): current list of tokens
        :param pair (tuple): pair of characters to merge
        :param new_token (str): the string representation of the new token
        """
        # list of new tokens
        new_tokens = []
        # target pair of tokens
        new_token = pair[0] + pair[1]
        # length of input tokens
        length = len(tokens)

        # loop through the token list and merge
        i = 0
        while (i < length - 1):
            curr = (tokens[i], tokens[i + 1])
            if (curr == pair):
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if (i == length - 1):
            new_tokens.append(tokens[i])
        # return list of new tokens
        return new_tokens

    def get_stats(self, tokens):
        """
        Returns the commonly occuring pair of characters in the list of input tokens
        
        :param tokens (list): list of input tokens
        """
        # create tuple to keep most common pair, dictionary of all pairs (str -> int)
        pairs_dict = {}

        # store token list length
        length = len(tokens)

        # initiate the loop
        i = 0
        while (i < length - 1):
            curr = (tokens[i], tokens[i+1])
            pairs_dict[curr] = pairs_dict.get(curr, 0) + 1
            i += 1
        
        return pairs_dict

def get_byte_to_char_mapping():
        """
        Produces the byte to char mapping dictionary
        """

        # define the 191 visible target characters (the characters we will consider when merging)
        ascii_bytes = list(range(33, 127))
        latin1_bytes = list(range(161, 256))

        visible_chars = ascii_bytes + latin1_bytes

        # define the 65 control bytes used to fill in our 256 mapping
        control_bytes_1 = list(range(33))
        control_bytes_2 = list(range(127, 161))

        control_bytes = control_bytes_1 + control_bytes_2

        # define the characters for the control bytes
        special_chars = [chr(i) for i in range(256, 256 + 67)]

        # define and construct the dictionary
        byte_to_char = {}

        for n in visible_chars:
            byte_to_char[n] = chr(n)

        for n, char in zip(control_bytes, special_chars):
            byte_to_char[n] = char
        
        # return output map
        return byte_to_char

