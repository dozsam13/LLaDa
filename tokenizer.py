class LLADACharacterTokenizer:
    def __init__(self, text):
        self.special_tokens = {
            '[MASK]': 0,
        }

        unique_chars = sorted(list(set(text)))

        self.char_to_id = self.special_tokens.copy()
        for i, char in enumerate(unique_chars):
            self.char_to_id[char] = len(self.special_tokens) + i

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.mask_token_id = self.special_tokens['[MASK]']



    def encode(self, text):
        return [self.char_to_id.get(char, self.mask_token_id) for char in text]

    def decode(self, token_ids):
        return ''.join([self.id_to_char.get(token_id, '[UNK]') for token_id in token_ids])
