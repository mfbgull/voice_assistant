#!/usr/bin/env python3
import os
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS

# ===============================
# 1. Speech-to-Text (STT) Setup
# ===============================
# Load the Whisper model (use "small" for a balance between speed and accuracy)
stt_model = WhisperModel("small", compute_type="int8")

def record_audio(duration=5, samplerate=16000):
    """
    Record audio from the default microphone.
    :param duration: Duration in seconds to record.
    :param samplerate: Sampling rate for recording.
    :return: A NumPy array containing the recorded audio.
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)

def transcribe_audio(audio, samplerate=16000):
    """
    Transcribe recorded audio using faster-whisper.
    :param audio: NumPy array of the recorded audio.
    :param samplerate: The audio sample rate.
    :return: The transcribed text.
    """
    segments, _ = stt_model.transcribe(audio, beam_size=5, language="en")
    transcription = " ".join(segment.text for segment in segments)
    print("Transcription:", transcription)
    return transcription

# ===============================
# 2. Integrate with Ollama LLM
# ===============================
def get_response_from_llm(prompt):
    """
    Send the transcribed prompt to a locally running LLM via Ollama.
    Ensure that your Ollama service is running and that you have the correct model loaded.
    :param prompt: The text prompt from the user.
    :return: The response text from the LLM.
    """
    try:
        model_name = "deepseek-r1:1.5b"
        if not ollama.is_model_available(model_name):
            print(f"Model '{model_name}' is not available. Please check your Ollama setup.")
            return "Sorry, the model is not available."
        
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        # Adjust the keys according to your Ollama API response format.
        response_text = response.get("message", {}).get("content", "")
        print("LLM Response:", response_text)
        return response_text
    except Exception as e:
        print("Error communicating with Ollama:", e)
        return "Sorry, I couldn't process your request."

# ===============================
# 3. Text-to-Speech (TTS) Setup
# ===============================
# Initialize the TTS model (using Coqui TTS)
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

def speak(text):
    """
    Convert text to speech using the TTS model and play the generated audio.
    :param text: The text to be spoken.
    """
    output_file = "response.wav"
    print("Speaking:", text)
    try:
        # Generate speech and save it to a file
        tts_model.tts_to_file(text=text, file_path=output_file)
        # Play the audio file (on Linux, 'aplay' is common; adjust if needed for your OS)
        os.system("aplay " + output_file)
    except Exception as e:
        print("Error during TTS:", e)

# ===============================
# 4. Main Loop with Both Input Modes
# ===============================
def main():
    print("Voice Assistant is running. You can choose your input method each time.")
    while True:
        try:
            print("\nSelect input mode:")
            print("1. Type your query")
            print("2. Speak your query")
            print("q. Quit")
            choice = input("Enter your choice (1/2/q): ").strip().lower()

            if choice == "1":
                # Text input mode
                user_input = input("You (text): ").strip()
                if not user_input:
                    print("No text entered. Please try again.")
                    continue
                prompt = user_input

            elif choice == "2":
                # Voice input mode
                audio = record_audio(duration=5)  # Adjust duration as needed.
                prompt = transcribe_audio(audio)
                if not prompt.strip():
                    print("No speech detected. Please try again.")
                    continue

            elif choice == "q":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please select 1, 2, or q.")
                continue

            # Process the prompt: Send to LLM and speak the response
            response_text = get_response_from_llm(prompt)
            speak(response_text)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print("An error occurred:", e)

if __name__ == "__main__":
    main()
