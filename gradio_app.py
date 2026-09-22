# gradio_app.py
from dotenv import load_dotenv

load_dotenv()  # must run before the other imports read any keys

import os
import tempfile

import gradio as gr

from brain_of_the_doctor import analyze_image_with_query, encode_image, guess_mime_type
from voice_of_the_doctor import text_to_speech
from voice_of_the_patient import transcribe_with_groq

VISION_MODEL = "qwen/qwen3.8-27b"
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")

LANGUAGES = {"English": "en", "Urdu": "ur"}

# English system prompt
system_prompt_en = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
With what I see, I think you have .... Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# Urdu system prompt
system_prompt_ur = """آپ کو ایک ماہر ڈاکٹر کی طرح برتاؤ کرنا ہے، یہ تربیتی مقاصد کے لیے ہے۔ 
جو کچھ میں دیکھ رہا ہوں، اس سے مجھے لگتا ہے کہ آپ کو .... جیسی حالت ہو سکتی ہے۔ 
کیا آپ کو طبی اعتبار سے کچھ غلط لگتا ہے؟ اگر مختلف تشخیص ہو سکتی ہیں تو ان کے علاج کے لیے کچھ تجاویز دیں۔
جواب مختصر رکھیں (زیادہ سے زیادہ دو جملے)، اور براہ کرم حقیقی مریض سے بات کرنے کے انداز میں جواب دیں۔
جواب کا آغاز براہ راست کریں، کوئی تمہید نہ ہو، اور کوئی نمبر یا خاص نشانات استعمال نہ کریں۔
جواب صرف اردو میں دیں۔
"""


def process_inputs(audio_filepath, image_filepath, language_choice):
    if not audio_filepath:
        raise gr.Error("Please record your question with the microphone first.")

    lang_code = LANGUAGES.get(language_choice, "en")

    # 1. Patient's voice -> text
    try:
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model=STT_MODEL,
            language=lang_code,
        )
    except Exception as e:
        raise gr.Error(f"Speech recognition failed: {e}")

    system_prompt = system_prompt_ur if lang_code == "ur" else system_prompt_en

    # 2. Image + question -> doctor's answer
    if image_filepath:
        try:
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model=VISION_MODEL,
                mime_type=guess_mime_type(image_filepath),
            )
        except Exception as e:
            raise gr.Error(f"Image analysis failed: {e}")
    else:
        doctor_response = (
            "No image provided for me to analyze"
            if lang_code == "en"
            else "مجھے دیکھنے کے لیے کوئی تصویر فراہم نہیں کی گئی"
        )

    # 3. Answer -> doctor's voice (unique file per request, so users don't clash)
    out_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    audio_out = text_to_speech(doctor_response, out_path, language=lang_code)

    return speech_to_text_output, doctor_response, audio_out


iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Patient's Voice Input"),
        gr.Image(type="filepath", label="Visual Analysis"),
        gr.Dropdown(list(LANGUAGES.keys()), value="English", label="Language"),
    ],
    outputs=[
        gr.Textbox(label="What patient asked"),
        gr.Textbox(label="What doctor responded with"),
        gr.Audio(type="filepath", label="Doctor's Voice Response"),
    ],
    title="YOUR PERSONALIZED DOCTOR",
    description=(
        "Talk to an AI doctor and get a response based on your voice and image input. "
        "For learning purposes only. This is not medical advice."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))