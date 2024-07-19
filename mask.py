import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import *
from transformers import AutoTokenizer, TFBertForMaskedLM
#from transformers import T5Model
import numpy as np
import os
#parts = tf.strings.split(file_path, os.path.sep)
# Pre-trained masked language model (install first)
MODEL = "bert-base-uncased"
#off-line
#MODEL = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)

#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#tokenizer = AutoTokenizer.from_pretrained(r"D:/VS/Shared/Python39_64/venv/PreTrainedModelOffline")
#model = AutoModelForSeq2SeqLM.from_pretrained(r"D:/VS/Shared/Python39_64/venv/PreTrainedModelOffline")
#tokenizer.save_pretrained("./Model")
#model.save_pretrained("./Model")

'''
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")

or programmatically install
python -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
'''

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype(r"E:\BigData\attention\assets\fonts\OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    #tokenizer = AutoTokenizer.from_pretrained(r"")
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    #mask_token_logits = result.logits[0, mask_token_index]
    mask_token_logits = result.logits[0,:, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens[0].flatten():
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    try:
        #if len(np.array(inputs.input_ids)[0])>1:
            #for ID in mask_token_id.added_tokens_decoder.keys():
                #if ID in np.array(inputs.input_ids)[0]:
        if mask_token_id in np.array(inputs.input_ids)[0]:
            return int(mask_token_id) 
        # TODO: Implement this function
        return None
    except:
        raise NotImplementedError

def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    try:
        # TODO: Implement this function
        if np.array(attention_score) >= 0 and np.array(attention_score) <= 1:
                AST = (round(np.array(attention_score)*255),round(np.array(attention_score)*255),round(np.array(attention_score)*255))
                return AST
        elif len(attention_score) == 3:
            AST = (int(round(attention_score[0]*255)),int(round(attention_score[1]*255)),int(round(attention_score[2]*255)))
            return AST
    except:
        raise NotImplementedError

def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    # TODO: Update this function to produce diagrams for all layers and heads.
    for i in range(np.shape(attentions)[0]):
        for k in range(np.shape(attentions)[3]):
            generate_diagram(
                i+1,
                k+1,
                tokens,
                attentions[i][0][k]
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"E:/BigData/attention/Attention_Layer{layer_number}_Head{head_number}.png")

if __name__ == "__main__":
    main()
