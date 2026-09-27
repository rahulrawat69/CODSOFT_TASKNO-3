"""
Image Captioning: CNN Encoder + RNN Decoder
---------------------------------------------
A minimal, readable implementation of the classic "Show and Tell"
architecture for image captioning:

  Image --> [Pretrained CNN encoder (ResNet-50)] --> feature vector
          --> [RNN decoder (LSTM)] --> generated caption, one word at a time

Requires: torch, torchvision, pillow
    pip install torch torchvision pillow --break-system-packages

Notes on running this yourself:
  - The encoder downloads pretrained ImageNet weights the first time it
    runs, so it needs an internet connection.
  - This file gives you a complete, working architecture and training
    loop. To get a captioning model that actually produces good captions,
    train it on a real dataset (e.g. MS-COCO Captions, Flickr8k/30k) for
    many epochs -- a few toy examples are not enough data to learn language.
  - A Transformer decoder (attention over image patches, like the "Show,
    Attend and Tell" or modern ViT+text-decoder models) is a drop-in swap
    for the DecoderRNN below; see the note at the bottom of the file.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter


# ---------------------------------------------------------------------------
# 1. Vocabulary: maps words <-> integer ids
# ---------------------------------------------------------------------------

class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.word2idx = {self.PAD: 0, self.START: 1, self.END: 2, self.UNK: 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def build(self, captions):
        """captions: list of raw caption strings."""
        counter = Counter()
        for caption in captions:
            counter.update(caption.lower().split())

        idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, caption, max_len=20):
        tokens = [self.START] + caption.lower().split() + [self.END]
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        ids = ids[:max_len]
        ids += [self.word2idx[self.PAD]] * (max_len - len(ids))
        return torch.tensor(ids)

    def decode(self, ids):
        words = []
        for i in ids:
            word = self.idx2word.get(int(i), self.UNK)
            if word == self.END:
                break
            if word not in (self.START, self.PAD):
                words.append(word)
        return " ".join(words)


# ---------------------------------------------------------------------------
# 2. Encoder: pretrained CNN -> fixed-size feature vector
# ---------------------------------------------------------------------------

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, train_cnn=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Drop the final classification layer; keep everything up to the
        # global-average-pooled feature vector (2048-dim for ResNet-50).
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Freeze the pretrained backbone by default -- we only train
        # a small linear projection into the decoder's embedding space.
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.set_grad_enabled(self.resnet.training and
                                     any(p.requires_grad for p in self.resnet.parameters())):
            features = self.resnet(images)          # (batch, 2048, 1, 1)
        features = features.reshape(features.size(0), -1)  # (batch, 2048)
        features = self.bn(self.fc(features))        # (batch, embed_size)
        return features


# ---------------------------------------------------------------------------
# 3. Decoder: RNN (LSTM) that generates a caption from the image feature
# ---------------------------------------------------------------------------

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Training-time forward pass (teacher forcing):
        the image feature is fed as the first "word" in the sequence,
        followed by the embedded ground-truth caption (minus the last token).
        """
        embeddings = self.embed(captions[:, :-1])                  # (batch, seq_len-1, embed)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)  # prepend image feature
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)                              # (batch, seq_len, vocab)
        return outputs

    def generate(self, features, vocab, max_len=20):
        """Greedy decoding at inference time: no ground-truth caption available."""
        result_ids = []
        inputs = features.unsqueeze(1)  # (batch=1, 1, embed_size)
        states = None

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))   # (batch, vocab)
            predicted = outputs.argmax(dim=1)            # (batch,)
            result_ids.append(predicted.item())

            if predicted.item() == vocab.word2idx[vocab.END]:
                break
            inputs = self.embed(predicted).unsqueeze(1)  # feed prediction back in

        return result_ids


# ---------------------------------------------------------------------------
# 4. Full model: wires the encoder and decoder together
# ---------------------------------------------------------------------------

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            feature = self.encoder(image.unsqueeze(0))
            ids = self.decoder.generate(feature, vocab, max_len)
        return vocab.decode(ids)


# ---------------------------------------------------------------------------
# 5. Image preprocessing (must match what the pretrained CNN expects)
# ---------------------------------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                          std=[0.229, 0.224, 0.225]),
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image_transform(image)


# ---------------------------------------------------------------------------
# 6. Toy training loop (illustrative -- swap in a real Dataset/DataLoader
#    over MS-COCO or Flickr8k for a model that actually generalizes)
# ---------------------------------------------------------------------------

def train_demo():
    # A handful of made-up (image path, caption) pairs, just to show the
    # training loop mechanics end to end. Replace with a real dataset.
    toy_data = [
        ("sample1.jpg", "a dog running on the beach"),
        ("sample2.jpg", "a plate of pasta with tomato sauce"),
        ("sample3.jpg", "a person riding a bicycle in the park"),
    ]

    vocab = Vocabulary(freq_threshold=1)
    vocab.build([caption for _, caption in toy_data])

    embed_size, hidden_size, num_layers = 256, 512, 1
    model = ImageCaptioningModel(embed_size, hidden_size, len(vocab), num_layers)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD])
    params = list(model.decoder.parameters()) + list(model.encoder.fc.parameters()) \
        + list(model.encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)

    print(f"Vocabulary size: {len(vocab)}")
    print("This demo skips real image loading (no sample images on disk) -- "
          "swap `load_image(path)` in for real data to train on actual pictures.")

    # Pseudocode for the real loop, once you have real images + captions:
    #
    # for epoch in range(num_epochs):
    #     for images, captions in dataloader:
    #         outputs = model(images, captions)
    #         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions.reshape(-1))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    return model, vocab


if __name__ == "__main__":
    train_demo()


# ---------------------------------------------------------------------------
# Extension: swapping in a Transformer decoder
# ---------------------------------------------------------------------------
# Replace DecoderRNN with a nn.TransformerDecoder (or a small custom
# decoder-only Transformer) that attends over image *patch* features
# instead of a single pooled vector:
#
#   - Encoder: drop the final avgpool/fc of the CNN, keeping the
#     (batch, 2048, 7, 7) spatial feature map; flatten to (batch, 49, 2048)
#     "tokens" and project to embed_size. This gives the decoder something
#     to attend over (like Show, Attend and Tell / modern ViT captioners).
#   - Decoder: standard causal self-attention over the caption tokens,
#     cross-attention over the encoded image tokens, then a linear+softmax
#     over the vocabulary -- the same interface (forward + generate) as
#     DecoderRNN above, so the rest of the pipeline stays unchanged.