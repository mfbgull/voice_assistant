#!/usr/bin/env python3
"""
Auto-activate virtual environment on Windows.
"""
import sys
import os

# Check if running in virtual environment
_VENV_PREFIX = os.environ.get('VIRTUAL_ENV', '')
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_SCRIPT_DIR, '.va-env', 'Scripts', 'python.exe')
_VENV_PYTHON_ALT = os.path.join(_SCRIPT_DIR, 'va-env', 'Scripts', 'python.exe')

# Find venv Python - use subprocess instead of execv for better Windows compatibility
if not _VENV_PREFIX and sys.platform == 'win32':
    for venv_python in [_VENV_PYTHON, _VENV_PYTHON_ALT]:
        if os.path.exists(venv_python):
            # Restart script with venv Python using subprocess
            import subprocess
            result = subprocess.run(
                [venv_python, __file__] + sys.argv[1:],
                cwd=_SCRIPT_DIR,
                env={**os.environ, 'VIRTUAL_ENV': _SCRIPT_DIR}
            )
            sys.exit(result.returncode)

"""
Voice Assistant - Enhanced Version v2.1 (KittenTTS Integration)
A user-friendly voice assistant with STT, LLM, TTS, and multilingual support.
Now featuring KittenTTS 80M model.
"""

import os
import sys
import json
import time
import subprocess
import threading
import signal
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict
from functools import lru_cache

# Try to import gTTS for multilingual fallback
try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    pass

# Try to import KittenTTS
try:
    from kittentts import KittenTTS
    import soundfile as sf
    _KITTENTTS_AVAILABLE = True
except ImportError:
    _KITTENTTS_AVAILABLE = False
    pass

# Try to import Anthropic for Claude models
try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    pass

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.columns import Columns
from rich import print as rprint
import shutil

# Initialize Rich console
console = Console()

# ===============================
# Configuration
# ===============================
CONFIG = {
    "stt_model": "base",
    "stt_compute_type": "int8",
    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
    "kitten_model": "KittenML/kitten-tts-mini-0.8",
    "default_model": "llama3",
    "recording_duration": 5,
    "samplerate": 16000,
    "ollama_timeout": 180,
    "max_history_messages": 10,
    "config_file": "va_config.json",
}

# Supported languages
LANGUAGES = {
    "en": {"name": "English", "code": "en", "tts": "en/ljspeech/tacotron2-DDC"},
    "ur": {"name": "Urdu", "code": "ur", "tts": "ur"},
    "ps": {"name": "Pashto", "code": "ps", "tts": "ps"},
}

@dataclass
class ConversationMessage:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AppConfig:
    """Application configuration."""
    model: str = "llama3"
    language: str = "en"
    input_mode: str = "text"
    tts_enabled: bool = True
    tts_engine: str = "whisper"  # 'whisper' (Coqui) or 'kitten'
    history_enabled: bool = True
    max_history: int = 10
    wake_word_enabled: bool = True
    wake_word: str = "ustaad"
    api_key: str = ""

# Global state
state = {
    "stt_model": None,
    "tts_model": None,
    "kitten_tts": None,
    "selected_model": None,
    "current_mode": "text",
    "current_language": "en",
    "ollama_running": None,
    "conversation_history": [],
    "config": AppConfig(),
    "running": True,
}

# ===============================
# Configuration Management
# ===============================
def get_config_path() -> Path:
    """Get path to config file."""
    return Path(CONFIG["config_file"])

def load_config() -> AppConfig:
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                import dataclasses
                valid_fields = {f.name for f in dataclasses.fields(AppConfig)}
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                return AppConfig(**filtered_data)
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Could not load config: {e}[/yellow]")
    return AppConfig()

def save_config(config: AppConfig):
    """Save configuration to file."""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Could not save config: {e}[/yellow]")

# ===============================
# Utility Functions
# ===============================
def clear_screen():
    """Clear the console screen."""
    console.clear()

def is_ollama_running():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_available_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().splitlines()
        if len(lines) <= 1:
            return []

        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []

def check_ollama_connection():
    """Verify Ollama is running and accessible."""
    if state["ollama_running"] is None:
        state["ollama_running"] = is_ollama_running()
    return state["ollama_running"]

def get_terminal_size():
    """Get terminal size for UI adjustments."""
    return shutil.get_terminal_size()

# ===============================
# Loading Animations
# ===============================
class LoadingSpinner:
    """Context manager for showing a loading spinner."""

    def __init__(self, message: str, spinner_name: str = "dots"):
        self.message = message
        self.spinner_name = spinner_name
        self._stop_event = threading.Event()
        self._thread = None
        self._live = None

    def _animate(self):
        """Background thread for spinner animation."""
        spinner = Spinner(self.spinner_name, text=self.message)
        with Live(spinner, console=console, refresh_per_second=20) as live:
            while not self._stop_event.is_set():
                time.sleep(0.05)
                live.update(Spinner(self.spinner_name, text=self.message))

    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)

def show_loading(message: str, duration: float | None = None):
    """Show a loading spinner for a duration or until context exits."""
    spinner = LoadingSpinner(message, "dots")
    spinner.__enter__()
    if duration is not None:
        time.sleep(duration)
        spinner.__exit__()
    return spinner

# ===============================
# Model Initialization (Lazy Loading)
# ===============================
def init_stt_model():
    """Initialize STT model with loading indicator."""
    if state["stt_model"] is None:
        with show_loading("Loading speech recognition model..."):
            state["stt_model"] = WhisperModel(
                CONFIG["stt_model"],
                compute_type=CONFIG["stt_compute_type"]
            )
    return state["stt_model"]

def init_tts_model():
    """Initialize Coqui TTS model with loading indicator."""
    if state["tts_model"] is None:
        with show_loading("Loading Whisper (Coqui) TTS engine...", duration=1):
            lang_code = state["current_language"]
            tts_model_path = CONFIG["tts_model"]
            state["tts_model"] = TTS(
                tts_model_path,
                gpu=torch.cuda.is_available()
            )
    return state["tts_model"]

def init_kitten_tts():
    """Initialize KittenTTS model with loading indicator."""
    if state["kitten_tts"] is None:
        if not _KITTENTTS_AVAILABLE:
            console.print("[red]Error: KittenTTS library not found.[/red]")
            return None
        with show_loading(f"Loading KittenTTS {CONFIG['kitten_model']}..."):
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("KittenML/kitten-tts-mini-0.8", "kitten_tts_mini_v0_8.onnx")
                # IMPORTANT: Use the nano-0.1 voices.npz because mini-0.8 has expressive voices (400, 256) 
                # but the mini-0.8 onnx model expects (1, 256). Nano's voices are (1, 256).
                voices_path = hf_hub_download("KittenML/kitten-tts-nano-0.1", "voices.npz")
                state["kitten_tts"] = KittenTTS(model_path, voices_path)
            except Exception as e:
                console.print(f"[red]Error initializing KittenTTS: {e}[/red]")
                return None
    return state["kitten_tts"]

# ===============================
# Wake Word Detection
# ===============================
WAKE_WORD_TIMEOUT = 30  # seconds to listen for wake word

def listen_for_wake_word(samplerate=16000, chunk_duration=3):
    """Listen for the wake word."""
    config_obj = state.get("config", AppConfig())
    wake_word = getattr(config_obj, 'wake_word', 'ustaad')

    if not wake_word:
        return True

    stt = init_stt_model()
    console.print(f"[cyan]👂 Listening for wake word '[bold]{wake_word}[/bold]'...[/cyan]")

    wake_panel = Panel(
        f"[bold cyan]🎧 Listening for '{wake_word}'...[/bold cyan]",
        title="Wake Word Detection",
        border_style="dim"
    )
    console.print(wake_panel)

    import time as time_module
    start_time = time_module.time()
    samples_per_chunk = int(chunk_duration * samplerate)

    try:
        while time_module.time() - start_time < WAKE_WORD_TIMEOUT:
            elapsed = int(time_module.time() - start_time)
            remaining = WAKE_WORD_TIMEOUT - elapsed
            console.print(f"[dim]Listening... ({remaining}s remaining)[/dim]", end="\r")

            try:
                audio = sd.rec(samples_per_chunk, samplerate=samplerate, channels=1, dtype=np.float32)
                sd.wait()
                audio = np.squeeze(audio)

                language = state["current_language"] or "en"
                segments, _ = stt.transcribe(audio, beam_size=5, language=language, vad_filter=True)
                text = " ".join(segment.text.strip() for segment in segments).lower()

                pattern = r'\b' + re.escape(wake_word) + r'\b'
                if re.search(pattern, text):
                    console.clear()
                    wake_detected_panel = Panel(
                        f"[bold green]🎉 Wake word '{wake_word}' detected![/bold green]\n\n"
                        "[dim]Now listening to your query...[/dim]",
                        title="Wake Word Detected",
                        border_style="green"
                    )
                    console.print(wake_detected_panel)
                    _play_beep(frequency=800, duration=0.1)
                    _play_beep(frequency=1200, duration=0.2)
                    return True

            except Exception:
                continue

        console.print("[yellow]⚠ Wake word listening timed out[/yellow]")
        return False
    except KeyboardInterrupt:
        return False

def _play_beep(frequency=440, duration=0.1):
    """Play a simple beep sound."""
    try:
        import winsound
        winsound.Beep(int(frequency), int(duration * 1000))
    except Exception:
        pass

# ===============================
# Speech-to-Text (STT)
# ===============================
def record_audio(duration=None, samplerate=None):
    """Record audio from microphone."""
    duration = duration or CONFIG["recording_duration"]
    samplerate = samplerate or CONFIG["samplerate"]
    console.print(f"[cyan]🎤 Recording for {duration} seconds...[/cyan]")
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        console.print(f"[red]Recording error: {e}[/red]")
        return None

def transcribe_audio(audio, samplerate=None, language=None):
    """Transcribe audio using Whisper."""
    if audio is None: return ""
    samplerate = samplerate or CONFIG["samplerate"]
    language = language or state["current_language"]
    stt = init_stt_model()
    with show_loading("Transcribing speech..."):
        try:
            segments, _ = stt.transcribe(audio, beam_size=5, language=language, vad_filter=True)
            transcription = " ".join(segment.text.strip() for segment in segments)
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
            return ""
    if transcription:
        console.print(f"[green]✓[/green] [bold]Heard:[/bold] {transcription}")
    else:
        console.print("[yellow]⚠ No speech detected[/yellow]")
    return transcription

# ===============================
# LLM Integration
# ===============================
def select_model():
    """Let user select an Ollama model."""
    models = get_available_models()
    all_models = []
    if _ANTHROPIC_AVAILABLE:
        all_models.extend(["claude-sonnet-4", "claude-sonnet-3.5", "claude-haiku-3"])
    all_models.extend(models)

    if not all_models:
        return CONFIG["default_model"]

    clear_screen()
    panel = Panel("[bold cyan]Select a Model[/bold cyan]\n\n" + "\n".join(f"[bold]{i}.[/bold] {m}" for i, m in enumerate(all_models, 1)), title="Model Selection", border_style="blue")
    console.print(panel)
    choice = Prompt.ask("Enter model number", choices=[str(i) for i in range(1, len(all_models) + 1)], default="1")
    selected = all_models[int(choice) - 1]
    return selected

def select_language():
    """Let user select output language."""
    clear_screen()
    panel = Panel("[bold cyan]Select Language[/bold cyan]\n\n" + "\n".join(f"[bold]{i}.[/bold] {LANGUAGES[code]['name']} ({code})" for i, code in enumerate(LANGUAGES.keys(), 1)), title="Language Selection", border_style="green")
    console.print(panel)
    choice = Prompt.ask("Enter language number", choices=[str(i) for i in range(1, len(LANGUAGES) + 1)], default="1")
    lang_code = list(LANGUAGES.keys())[int(choice) - 1]
    state["current_language"] = lang_code
    state["tts_model"] = None
    return lang_code

def select_tts_engine():
    """Let user select TTS engine."""
    clear_screen()
    panel = Panel(
        """[bold cyan]Select TTS Engine[/bold cyan]

[bold]1.[/bold] Whisper (Coqui TTS) - High quality, slower
[bold]2.[/bold] KittenTTS (80M Model) - Fast, realistic, CPU optimized""",
        title="TTS Selection",
        border_style="magenta"
    )
    console.print(panel)
    choice = Prompt.ask("Enter choice", choices=["1", "2"], default="2")
    engine = "whisper" if choice == "1" else "kitten"
    state["config"].tts_engine = engine
    save_config(state["config"])
    return engine

def select_wake_word():
    """Let user select wake word."""
    clear_screen()
    panel = Panel(
        """[bold cyan]Wake Word Setup[/bold cyan]

Options:
1. 'Ustaad'
2. 'Hey Assistant'
3. 'Computer'
4. Custom
5. Disable""",
        title="Wake Word Configuration",
        border_style="green"
    )
    console.print(panel)
    choice = Prompt.ask("Enter choice", choices=["1", "2", "3", "4", "5"], default="1")
    if choice == "1": wake_word = "ustaad"
    elif choice == "2": wake_word = "hey assistant"
    elif choice == "3": wake_word = "computer"
    elif choice == "4": wake_word = Prompt.ask("Enter wake word").strip().lower()
    else: wake_word = ""
    state["config"].wake_word = wake_word
    state["config"].wake_word_enabled = (choice != "5")
    return wake_word

def get_llm_response(prompt: str) -> str:
    """Get response from Ollama."""
    model = state["selected_model"]
    if not model or not check_ollama_connection(): return "Error: Ollama not ready."
    full_response = []
    console.print(f"[cyan]🤖 Thinking with [bold]{model}[/bold]...[/cyan]")
    try:
        messages = [{"role": m.role, "content": m.content} for m in state["conversation_history"]]
        lang_name = LANGUAGES.get(state["current_language"], {}).get("name", "English")
        messages.insert(0, {"role": "system", "content": f"Respond in {lang_name}."})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(model=model, messages=messages, stream=True)
        with console.status("[bold cyan]Generating...", spinner="dots"):
            for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                if content: full_response.append(content)
        return "".join(full_response).strip()
    except Exception as e: return f"Error: {e}"

# ===============================
# Text-to-Speech (TTS)
# ===============================
def clean_text_for_tts(text: str) -> str:
    """Basic cleaning for TTS."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII
    return " ".join(text.split()).strip()

def speak(text: str):
    """Convert text to speech and play it."""
    if not text or not state["config"].tts_enabled: return
    text = clean_text_for_tts(text)
    if len(text) < 2: return
    
    output_file = "response.wav"
    console.print(f"[yellow]🔊 Speaking ({state['config'].tts_engine})...[/yellow]")

    if state["config"].tts_engine == "kitten" and _KITTENTTS_AVAILABLE:
        _speak_kitten(text, output_file)
    else:
        _speak_whisper(text, output_file)

def _speak_whisper(text, output_file):
    """Coqui TTS (Whisper labeled)."""
    if state["current_language"] != "en" and _GTTS_AVAILABLE:
        _speak_gtts(text, output_file)
        return
    tts = init_tts_model()
    try:
        with console.status("[bold yellow]Generating Whisper TTS...", spinner="dots"):
            tts.tts_to_file(text=text, file_path=output_file)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]TTS Error: {e}[/red]")

def _speak_kitten(text, output_file):
    """KittenTTS."""
    kitten = init_kitten_tts()
    if not kitten:
        _speak_whisper(text, output_file)
        return
    try:
        with console.status("[bold yellow]Generating KittenTTS...", spinner="dots"):
            # The library generate method might work now with the right voices.npz
            audio = kitten.generate(text, voice='expr-voice-2-m')
            sf.write(output_file, audio, 24000)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]KittenTTS Error: {e}[/red]")
        _speak_whisper(text, output_file)

def _speak_gtts(text, output_file):
    """Google TTS."""
    try:
        lang = state["current_language"]
        tts = gTTS(text=text, lang=lang)
        tts.save(output_file)
        _play_audio(output_file)
    except Exception: pass

def _play_audio(file):
    """Play audio."""
    if sys.platform == "win32":
        subprocess.run(['cmd', '/c', 'start', '/B', '/WAIT', '', file], shell=False)
    elif sys.platform == "darwin":
        subprocess.run(['afplay', file])
    else:
        subprocess.run(['aplay', file])

# ===============================
# Main Loop
# ===============================
def display_banner():
    clear_screen()
    config = state["config"]
    banner = Panel(
        Text(f"""🎙️ VOICE ASSISTANT v2.1
Engine: {config.tts_engine.upper()}
Mode: {state['current_mode'].upper()}""", justify="center", style="bold cyan"),
        title="Welcome",
        border_style="blue"
    )
    console.print(banner)

def main():
    state["config"] = load_config()
    display_banner()
    
    # Selection Flow
    state["selected_model"] = select_model()
    state["current_language"] = select_language()
    state["config"].tts_engine = select_tts_engine()
    state["current_mode"] = "voice" if Prompt.ask("Input Mode", choices=["1", "2"], default="2") == "2" else "text"
    wake = select_wake_word()
    
    save_config(state["config"])

    while state["running"]:
        display_banner()
        if state["current_mode"] == "text":
            prompt = Prompt.ask("\n[bold green]You[/bold green]")
        else:
            if state["config"].wake_word_enabled:
                if not listen_for_wake_word(): continue
            audio = record_audio()
            prompt = transcribe_audio(audio)
        
        if not prompt: continue
        if prompt.lower() in ["q", "quit"]: break
        
        response = get_llm_response(prompt)
        console.print(Panel(f"[bold]Assistant:[/bold]\n{response}", border_style="magenta"))
        
        state["conversation_history"].append(ConversationMessage("user", prompt))
        state["conversation_history"].append(ConversationMessage("assistant", response))
        
        speak(response)
        console.print("\n[dim]Press Enter to continue...[/dim]")
        input()

if __name__ == "__main__":
    main()
