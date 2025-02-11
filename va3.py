#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS

# For a fancier UI using Rich
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Initialize the Rich console.
console = Console()

# -------------------------------
# Global: Selected Model (set at startup)
# -------------------------------
selected_model = None

# -------------------------------
# 1. Model Selection Function
# -------------------------------
def select_model():
    """
    Runs 'ollama list' to show available models and lets the user choose one by number.
    Returns the chosen model name as a string.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models_output = result.stdout.strip()
        if not models_output:
            console.print("[red]No models found via 'ollama list'. Defaulting to 'llama3'.[/red]")
            return "llama3"
        # Assume each nonempty line represents one model.
        models = [line.strip() for line in models_output.splitlines() if line.strip()]
        if not models:
            console.print("[red]No models parsed from 'ollama list'. Defaulting to 'llama3'.[/red]")
            return "llama3"
        # Build a numbered list of models.
        models_str = "\n".join(f"{i}. {model}" for i, model in enumerate(models, start=1))
        panel = Panel(models_str, title="Available Models (ollama list)", border_style="blue")
        console.print(panel)
        choices = [str(i) for i in range(1, len(models) + 1)]
        choice = Prompt.ask("Select a model by number", choices=choices, default="1")
        chosen_model = models[int(choice) - 1]
        return chosen_model
    except Exception as e:
        console.print(f"[red]Error listing models:[/red] {e}")
        return "llama3"

# -------------------------------
# 2. Speech-to-Text (STT) Setup
# -------------------------------
# Load the Whisper model (using "small" for a balance between speed and accuracy)
stt_model = WhisperModel("small", compute_type="int8")

def transcribe_audio(audio, samplerate=16000):
    """
    Transcribes the given audio (NumPy array) using faster-whisper.
    Returns the transcribed text.
    """
    segments, _ = stt_model.transcribe(audio, beam_size=5, language="en")
    transcription = " ".join(segment.text for segment in segments)
    return transcription

def record_single(chunk_duration=10, samplerate=16000):
    """
    Records a single audio clip for the specified duration (10 seconds by default),
    transcribes it, and returns the transcription.
    """
    console.print(f"[cyan]Recording for {chunk_duration} seconds...[/cyan]")
    audio = sd.rec(int(chunk_duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    audio = np.squeeze(audio)
    transcription = transcribe_audio(audio, samplerate)
    console.print(f"[green]Transcription:[/green] {transcription}")
    return transcription

# -------------------------------
# 3. Integrate with Ollama LLM
# -------------------------------
def get_response_from_llm(prompt):
    """
    Sends the prompt to the locally running LLM via Ollama using the selected model.
    Returns the response text.
    """
    try:
        response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": prompt}])
        response_text = response.get("message", {}).get("content", "")
        return response_text
    except Exception as e:
        console.print(f"[red]Error communicating with Ollama:[/red] {e}")
        return "Sorry, I couldn't process your request."

# -------------------------------
# 4. Text-to-Speech (TTS) Setup
# -------------------------------
# Initialize the TTS model (using Coqui TTS)
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())

def speak(text):
    """
    Converts text to speech using the TTS model and plays the generated audio.
    """
    output_file = "response.wav"
    console.print(f"[yellow]Speaking:[/yellow] {text}")
    try:
        tts_model.tts_to_file(text=text, file_path=output_file)
        # Play the audio (using 'aplay' for Linux; adjust as needed for your OS)
        os.system("aplay " + output_file)
    except Exception as e:
        console.print(f"[red]Error during TTS:[/red] {e}")

# -------------------------------
# 5. Enhanced UI Functions
# -------------------------------
def display_ui(current_mode):
    """
    Clears the console and displays a professional UI with the current input mode
    and selected model.
    """
    console.clear()
    header = Panel(
        "[bold cyan]Voice Assistant[/bold cyan]\n"
        f"Current Input Mode: [bold green]{current_mode.upper()}[/bold green]\n"
        f"Using Model: [bold magenta]{selected_model}[/bold magenta]\n\n"
        "Press [bold]M[/bold] at any time to change input mode.",
        title="Welcome", border_style="blue", expand=False
    )
    console.print(header)
    console.rule()

def input_mode_menu():
    """
    Presents a menu to select the input mode.
    Returns the selected mode as a string: 'text' or 'voice'.
    """
    display_ui("Select Mode")
    console.print("[bold]Input Mode Menu[/bold]\n1. Text Mode\n2. Voice Mode")
    choice = Prompt.ask("Enter 1 or 2", choices=["1", "2"], default="1")
    return "text" if choice == "1" else "voice"

# -------------------------------
# 6. Main Loop
# -------------------------------
def main():
    global selected_model
    # Step 1: Select the desired model from the local list.
    selected_model = select_model()
    
    # Step 2: Ask for the input mode once at startup.
    current_mode = input_mode_menu()

    while True:
        try:
            # Display the updated UI with the current input mode and selected model.
            display_ui(current_mode)
            console.print("[bold]Press [M] to change input mode, or just press Enter to continue.[/bold]")
            key = input(">> ").strip().lower()
            if key == "m":
                current_mode = input_mode_menu()
                continue  # Redisplay the UI and menu

            # Process the query based on the selected input mode.
            if current_mode == "text":
                prompt_text = Prompt.ask("You (text)")
            else:
                # Voice mode: record a single 10-second clip.
                prompt_text = record_single(chunk_duration=10)
            
            if not prompt_text:
                console.print("[red]No input detected. Please try again.[/red]")
                continue

            # Display the user's prompt in a separate panel.
            console.print(Panel(f"[bold]Your Input:[/bold] {prompt_text}", border_style="green"))
            
            # Get LLM response.
            response_text = get_response_from_llm(prompt_text)
            console.print(Panel(f"[bold]LLM Response:[/bold] {response_text}", border_style="magenta"))
            
            # Speak the LLM's response.
            speak(response_text)
            
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        except Exception as e:
            console.print(f"[red]An error occurred:[/red] {e}")

if __name__ == "__main__":
    main()
