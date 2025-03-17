from safetensors.torch import save_model
import torch
from torch import nn

from model import MiniHyena
from tokenizer import AsciiTokenizer


# This small excerpt of the Bee Movie script is used for the sole purpose of research
sentence = "According to all known laws of aviation, there is no way a bee should be able to fly."

# parameters
ARCHITECTURE = "aah"
D_MODEL = 512
L_MAX = 128
BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 20
N_TESTS = 1000


def main() -> None:
    # tokenize the input
    tokenizer = AsciiTokenizer()
    tokens = tokenizer.tokenize(sentence)
    tokens = torch.tensor(tokens).repeat(BATCH_SIZE, 1).to(torch.int64)

    torch.manual_seed(42)

    # create the model
    model = MiniHyena(d_model=D_MODEL, l_max=L_MAX, vocab_size=512, blocks=ARCHITECTURE)

    # create model input and training parameters
    x = {"input": tokens}
    y = dict()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = EPOCHS

    # train the model
    for epoch in range(n_epochs):
        loss, output = model.batch(x, y, loss_fn, optimizer)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item()}")

    # test the model
    n_tests = N_TESTS

    total_chars = 0
    total_chars_good = 0
    final_chars = 0
    final_chars_good = 0

    for i in range(n_tests):
        # select a random 10-character substring from the sentence
        start_idx = torch.randint(0, len(sentence) - 11, (1,)).item()
        test_sentence = sentence[start_idx:start_idx + 10]

        # tokenize the test sentence
        test_tokens = tokenizer.tokenize(test_sentence)
        test_tokens = torch.tensor(test_tokens).unsqueeze(0).to(torch.int64)
        test_output = model(test_tokens)

        # detokenize the output and compare it to the original 10-character substring
        result_seq = tokenizer.detokenize(test_output[0].argmax(dim=-1))
        results_list = list(zip(result_seq, sentence[start_idx + 1:start_idx + 11]))
        good = [r == t for r, t in results_list]

        # count correct characters
        total_chars += len(results_list)
        total_chars_good += sum(good)
        final_chars += 1
        final_chars_good += good[-1]

    print(total_chars_good, total_chars, final_chars_good, final_chars)

    # save the model
    save_model(model, "bee_movie.safetensors")


if __name__ == "__main__":
    main()
