
import wave
import pyaudio
import whisper
from google.cloud import texttospeech
import wave
import torch
import os
from openai import OpenAI
from google.cloud import texttospeech
client = OpenAI(base_url="https://localhost:5317",api_key="not-needed")

chat_log = "chatbot.txt"


en_ckpt_base = "checkpoints/base_speakers/EN"
ckpt_converter = "checkpoints/converter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

def process_and_play(prompt, audio_file_path):
    client_speech = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=prompt)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Journey-D" 
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, 
        effects_profile_id=["small-bluetooth-speaker-class-device"],  
        pitch=0,  
        speakingRate=1  
    )

    response = client_speech.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    with open(audio_file_path, 'wb') as output:
        output.write(response.audio_content)
        print("Audio content written to file", audio_file_path)
    play_audio(audio_file_path)

def play_audio(audio_file_path):
    wf = wave.open(audio_file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()
def transcribe_with_whisper(audio_file_path):
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_file_path)
    return result["text"]

def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    frames = []
    print("Recording...")
    for i in range(0, int(16000 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording done.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def model_streamed(user_input,system_message,conversation_history,bot_name):
    messages = [{"role":"system","content":system_message},conversation_history,{"role":"user","content":user_input}]
    temperature = 1
    
    streamed_completion = client.chat.completions.create(model="gpt-3.5-turbo",messages=messages,stream=True)
    full_response = ""
    line_buffer = ""
    
    with open(chat_log, "a") as file:
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                line_buffer += delta_content
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(f"{bot_name}:",line)
                        full_response += line + "\n"
                        file.write(f"{bot_name}: {line}\n")
                    line_buffer = lines[-1]
        if line_buffer:
            print(f"{bot_name}:",line_buffer)
            full_response += line_buffer
            file.write(f"{bot_name}: {line_buffer}\n")
    return full_response

