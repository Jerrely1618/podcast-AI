import argparse
import functions as f
import os

parser = argparse.ArgumentParser(description="Run the AI chatbot")
parser.add_argument("--audio_file", default="tmp_recording.wav", help="The audio file to record and process")
args = parser.parse_args()
def open_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()
def chatbot():
    conversation_history = []
    system_message = open_file(f.chat_log)
    while True:
        f.record_audio(args.audio_file)
        user_input = f.transcribe_with_whisper(args.audio_file)
        os.remove(args.audio_file)
        
        if user_input.lower() == "exit":
            break
        print("User:", user_input)
        conversation_history.append({"role":"user", "content":user_input})
        print("Michael:", system_message)
        chatbot_response = f.model_streamed(user_input, system_message, conversation_history, "Michael")
        conversation_history.append({"role":"system", "content":chatbot_response})
        
        prompt2 = chatbot_response
        style = "default"
        audio_file_path2 = "C:/Users/Owner/OneDrive/Desktop/podcast-AI/out.mp3"
        f.process_and_play(prompt2, style, audio_file_path2)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
    
chatbot()
