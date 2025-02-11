#!/usr/bin/env python3

import os
import subprocess
import time
import sounddevice as sd
import numpy as np
import queue
import ollama
import speech_recognition as sr
from rich.console import Console
from rich.table import Table
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from TTS.api import TTS

# Initialize Console UI
console = Console()

# Global Variables
q = queue.Queue()
recording_duration = 10  # Record for 10 seconds
selected_model = None  # Store user's selected LLM model
input_mode = None  # Store input mode (1 for text, 2 for voice)

# Display Banner
def display_banner():
    console.clear()
    banner_text = Text("VOICE ASSISTANT", style="bold cyan", justify="center")
    developer_text = Text("Developed by FAWAD using AI", style="bold yellow", justify="center")
    
    # Print top-right corner name
    console.print(Align.right(Text("FAWAD", style="bold magenta")))
    
    console.print(Align.center(banner_text))
    console.print(Align.center(developer_text))
    console.print("\n" + "=" * 50)

# List Available Ollama Models
def get_local_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")

        if len(lines) <= 1:
            return []

        models = [line.split()[0] for line in lines[1:]]
        return models

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        return []

# Prompt User to Select a Model
def select_model():
    global selected_model
    models = get_local_models()
    
    if not models:
        console.print("[red]No local models found! Please install an Ollama model first.[/red]")
        return None

    console.print("\n[bold cyan]Select a Model:[/bold cyan]")
    table = Table(title="Available Models", show_lines=True)
    table.add_column("Option", justify="center", style="bold yellow")
    table.add_column("Model Name", justify="left", style="bold green")

    for i, model in enumerate(models, start=1):
        table.add_row(str(i), model)

    console.print(table)
    choice = console.input("[bold green]Enter the model number: [/bold green]")
    
    try:
        selected_model = models[int(choice) - 1]
        console.print(f"[bold cyan]Selected Model:[/bold cyan] {selected_model}")
    except (IndexError, ValueError):
        console.print("[red]Invalid selection! Try again.[/red]")
        select_model()

# Record Audio for 10 Seconds
def record_audio():
    console.print("[bold yellow]Recording... Speak now! (Max 10 sec)[/bold yellow]")

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)
            console.print("[green]Recording finished![/green]")
            return audio
        except sr.WaitTimeoutError:
            console.print("[red]No speech detected, try again![/red]")
            return None

# Convert Speech to Text
def speech_to_text(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        console.print("[red]Sorry, could not understand the audio.[/red]")
        return None
    except sr.RequestError:
        console.print("[red]Error with speech recognition service.[/red]")
        return None

# Generate Response using Ollama
def get_ai_response(user_input):
    if not selected_model:
        console.print("[red]No model selected! Please choose a model first.[/red]")
        return "No model selected."

    console.print("[bold blue]Thinking...[/bold blue]")
    response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": user_input}])
    return response["message"]["content"]

# Convert Text to Speech
def text_to_speech(text):
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2")
    tts.tts_to_file(text=text, file_path="response.wav")
    
    os.system("aplay response.wav")  # Play the generated speech

# Prompt User for Input Mode (Text or Voice)
def select_input_mode():
    global input_mode
    console.print("\n[bold cyan]Select Input Mode:[/bold cyan]")
    table = Table(title="Input Modes", show_lines=True)
    table.add_column("Option", justify="center", style="bold yellow")
    table.add_column("Mode", justify="left", style="bold green")

    table.add_row("1", "Text Input")
    table.add_row("2", "Voice Input")

    console.print(table)
    choice = console.input("[bold green]Enter mode number: [/bold green]")

    if choice in ["1", "2"]:
        input_mode = int(choice)
        console.print(f"[bold cyan]Selected Input Mode:[/bold cyan] {'Text' if input_mode == 1 else 'Voice'}")
    else:
        console.print("[red]Invalid selection! Try again.[/red]")
        select_input_mode()

# Main Loop
def main():
    display_banner()
    select_model()
    select_input_mode()

    while True:
        console.print("\n[bold yellow]Press 'm' to change input mode, or 'q' to quit.[/bold yellow]")

        if input_mode == 1:  # Text Mode
            user_input = console.input("[bold green]You:[/bold green] ")
            if user_input.lower() == "q":
                break
            if user_input.lower() == "m":
                select_input_mode()
                continue
            
            response = get_ai_response(user_input)
            console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")
            text_to_speech(response)

        elif input_mode == 2:  # Voice Mode
            audio = record_audio()
            if not audio:
                continue

            user_input = speech_to_text(audio)
            if not user_input:
                continue

            console.print(f"[bold green]You:[/bold green] {user_input}")
            response = get_ai_response(user_input)
            console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")
            text_to_speech(response)

if __name__ == "__main__":
    main()
