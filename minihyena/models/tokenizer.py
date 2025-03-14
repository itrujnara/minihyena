import numpy as np

class AsciiTokenizer():
    def tokenize(self, text):
        return list(np.frombuffer(text.encode(), dtype=np.uint8))
    
    def tokenize_batch(self, texts):
        return [self.tokenize(text) for text in texts]

    def detokenize(self, tokens):
        return "".join([chr(i) for i in tokens])
