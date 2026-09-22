# AI_Multilingial_Doctor
AI DOCTOR VOICE + VISION

A learning project: record a question, upload a photo (for example a skin issue), and the app answers in a doctor-like voice. Supports English and Urdu.

Disclaimer: This is for learning purposes only. It is not medical advice. Always consult a real doctor.

How it works
-> Speech-to-text: Groq-hosted Whisper transcribes your voice.
-> Vision + language: a multimodal Groq model looks at the image and your question.
-> Text-to-speech: ElevenLabs speaks the answer (falls back to gTTS if ElevenLabs is unavailable).
-> UI: Gradio.
