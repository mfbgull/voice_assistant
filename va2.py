#!/usr/bin/env python3
import os
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS

# For a fancier UI, we'll use the rich library.
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Initialize the Rich console.
console = Console()

# ===============================
# 1. Speech-to-Text (STT) Setup
# ===============================
# Load the Whisper model (using "small" for a balance between speed and accuracy)
stt_model = WhisperModel("small", compute_type="int8")

def record_audio(duration=5, samplerate=16000):
    """
    Record audio from the default microphone.
    """
    console.print(f"[cyan]Recording for {duration} seconds...[/cyan]")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)

def transcribe_audio(audio, samplerate=16000):
    """
    Transcribe recorded audio using faster-whisper.
    """
    segments, _ = stt_model.transcribe(audio, beam_size=5, language="en")
    transcription = " ".join(segment.text for segment in segments)
    console.print(f"[green]Transcription:[/green] {transcription}")
    return transcription

# ===============================
# 2. Integrate with Ollama LLM
# ===============================
def get_response_from_llm(prompt):
    """
    Send the prompt to the locally running LLM via Ollama.
    Ensure your Ollama service is running and the proper model (e.g., "llama3") is loaded.
    """
    try:
        response = ollama.chat(model="deepseek-coder", messages=[{"role": "user", "content": prompt}])
        response_text = response.get("message", {}).get("content", "")
        console.print(f"[magenta]LLM Response:[/magenta] {response_text}")
        return response_text
    except Exception as e:
        console.print(f"[red]Error communicating with Ollama:[/red] {e}")
        return "Sorry, I couldn't process your request."

# ===============================
# 3. Text-to-Speech (TTS) Setup
# ===============================
# Initialize the TTS model (using Coqui TTS)
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

def speak(text):
    """
    Convert text to speech using the TTS model and play the generated audio.
    """
    output_file = "response.wav"
    console.print(f"[yellow]Speaking:[/yellow] {text}")
    try:
        tts_model.tts_to_file(text=text, file_path=output_file)
        # On Linux, use 'aplay'; on Windows you might use 'start' and on macOS 'afplay'.
        os.system("aplay " + output_file)
    except Exception as e:
        console.print(f"[red]Error during TTS:[/red] {e}")

# ===============================
# 4. UI Functions
# ===============================
def display_ui(current_mode):
    """
    Clears the console and displays a fancy UI panel with the current input mode.
    """
    console.clear()
    banner = Panel(
        f"[bold cyan]Voice Assistant[/bold cyan]\n\nCurrent Input Mode: [bold green]{current_mode.upper()}[/bold green]\n\n"
        "Press [bold]m[/bold] at any time to change the input mode.",
        title="Welcome", expand=False, border_style="blue"
    )
    console.print(banner)

def input_mode_menu():
    """
    Presents a menu to select the input mode.
    Returns the selected mode as a string: 'text' or 'voice'.
    """
    display_ui("Select Mode")
    console.print("[bold]Input Mode Menu[/bold]\n1. Text Mode\n2. Voice Mode")
    choice = Prompt.ask("Enter 1 or 2", choices=["1", "2"], default="1")
    if choice == "1":
        return "text"
    else:
        return "voice"

# ===============================
# 5. Main Loop
# ===============================
def main():
    # Ask for the input mode once at startup.
    current_mode = input_mode_menu()

    while True:
        try:
            # Show the fancy UI with the current mode.
            display_ui(current_mode)
            console.print("[bold]Press [m] to change input mode, or simply press Enter to continue.[/bold]")
            key = input(">> ").strip().lower()
            if key == "m":
                # Change input mode if the user presses 'm'
                current_mode = input_mode_menu()
                continue  # Re-display the UI and menu

            # Get the prompt based on the current input mode.
            if current_mode == "text":
                prompt = input("You (text): ").strip()
            else:
                # Voice mode
                audio = record_audio(duration=5)  # Adjust duration if needed
                prompt = transcribe_audio(audio)
            
            if not prompt:
                console.print("[red]No input detected. Please try again.[/red]")
                continue

            # Send the prompt to the LLM and speak the response.
            response_text = get_response_from_llm(prompt)
            speak(response_text)

            # Optional: Wait for user confirmation to continue.
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        except Exception as e:
            console.print(f"[red]An error occurred:[/red] {e}")

if __name__ == "__main__":
    main()
