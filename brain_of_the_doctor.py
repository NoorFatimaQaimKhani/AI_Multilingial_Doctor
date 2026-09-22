# brain_of_the_doctor.py
import base64
import mimetypes

from groq import Groq


def encode_image(image_path):
    """Read an image file and return it as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_query(query, model, encoded_image, mime_type="image/jpeg"):
    """Send the patient's question plus the image to a multimodal Groq model."""
    client = Groq()  # reads GROQ_API_KEY from the environment
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(messages=messages, model=model, max_completion_tokens=512,
        reasoning_effort="none",)
    return chat_completion.choices[0].message.content


def guess_mime_type(image_path):
    return mimetypes.guess_type(image_path)[0] or "image/jpeg"