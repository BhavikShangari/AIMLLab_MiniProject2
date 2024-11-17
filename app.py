import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransTokenizer.IndicTransTokenizer.processor import IndicProcessor
from transformers import pipeline

@st.cache_resource
def load_model():
    image_caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    image_caption_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_caption_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    model_name = "ai4bharat/indictrans2-en-indic-1B"
    translation_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    indic_processor = IndicProcessor(inference=True)

    audio_generator = pipeline("text-to-speech", model="suno/bark-small", device=device)
    return image_caption_model, image_caption_tokenizer, image_caption_image_processor, translation_tokenizer, translation_model, indic_processor, audio_generator

image_caption_model, image_caption_tokenizer, image_caption_image_processor, translation_tokenizer, translation_model, indic_processor, audio_generator = load_model()

src_lang, tgt_lang = "eng_Latn", "hin_Deva"
# Title for the app
st.title("Upload an Image")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    gen_text = image_caption_tokenizer.decode(image_caption_model.generate(pixel_values  = image_caption_image_processor(image, return_tensors="pt").to(device).pixel_values).cpu()[0], skip_special_tokens=True)
    # Display the image
    input_sentences = [
        gen_text
    ]
    batch = indic_processor.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    inputs = translation_tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

# Decode the generated tokens into text
    with translation_tokenizer.as_target_tokenizer():
        generated_tokens = translation_tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    # Postprocess the translations, including entity replacement
    translations = indic_processor.postprocess_batch(generated_tokens, lang=tgt_lang)
    st.image(image, caption=translations[0], use_column_width=True)

    output = audio_generator(translations)
    st.write('Audio Output')
    st.audio(output[0]['audio'], format="audio/mp3", sample_rate=output[0]['sampling_rate'])
