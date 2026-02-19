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
Voice Assistant - Enhanced Version v2.0
A user-friendly voice assistant with STT, LLM, TTS, and multilingual support.
"""

import os
import sys
import json
import time
import subprocess
import threading
import signal
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
    # Silence warning during import
    pass

# Try to import Anthropic for Claude models
try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    # Silence warning during import
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
from rich.console import Console
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
    history_enabled: bool = True
    max_history: int = 10
    wake_word_enabled: bool = True
    wake_word: str = "ustaad"
    api_key: str = ""

# Global state
state = {
    "stt_model": None,
    "tts_model": None,
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
                return AppConfig(**data)
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

        lines = result.stdout.strip().split("\n")
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
    """Initialize TTS model with loading indicator."""
    if state["tts_model"] is None:
        with show_loading("Loading text-to-speech engine...", duration=1):
            # Use language-specific TTS model
            lang_code = state["current_language"]
            tts_model_path = CONFIG["tts_model"]
            if lang_code != "en":
                # For non-English, try to use a multilingual model or fall back
                tts_model_path = CONFIG["tts_model"]

            state["tts_model"] = TTS(
                tts_model_path,
                gpu=torch.cuda.is_available()
            )
    return state["tts_model"]

# ===============================
# Wake Word Detection
# ===============================
WAKE_WORD_TIMEOUT = 30  # seconds to listen for wake word

def listen_for_wake_word(samplerate=16000, chunk_duration=3):
    """
    Listen for the wake word.
    Returns True if wake word detected, False if timeout or error.
    """
    config_obj = state.get("config", AppConfig())
    wake_word = getattr(config_obj, 'wake_word', 'ustaad')

    if not wake_word:
        return True  # No wake word configured, skip detection

    stt = init_stt_model()
    console.print(f"[cyan]👂 Listening for wake word '[bold]{wake_word}[/bold]'...[/cyan]")

    # Create a status panel for wake word detection
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
            # Update status
            elapsed = int(time_module.time() - start_time)
            remaining = WAKE_WORD_TIMEOUT - elapsed

            # Record a chunk
            console.print(f"[dim]Listening... ({remaining}s remaining)[/dim]", end="\r")

            try:
                audio = sd.rec(samples_per_chunk, samplerate=samplerate, channels=1, dtype=np.float32)
                sd.wait()
                audio = np.squeeze(audio)

                # Transcribe the chunk - use current language or default to English
                language = state["current_language"] or "en"
                segments, _ = stt.transcribe(
                    audio,
                    beam_size=5,
                    language=language,
                    vad_filter=True
                )
                text = " ".join(segment.text.strip() for segment in segments).lower()

                # Check for wake word with word boundary matching to reduce false positives
                # Create a pattern that matches the wake word as a whole word or at sentence boundaries
                pattern = r'\b' + re.escape(wake_word) + r'\b'
                if re.search(pattern, text):
                    # Wake word detected!
                    console.clear()
                    wake_detected_panel = Panel(
                        f"[bold green]🎉 Wake word '{wake_word}' detected![/bold green]\n\n"
                        "[dim]Now listening to your query...[/dim]",
                        title="Wake Word Detected",
                        border_style="green"
                    )
                    console.print(wake_detected_panel)

                    # Play a sound to indicate wake word detected
                    _play_beep(frequency=800, duration=0.1)
                    _play_beep(frequency=1200, duration=0.2)

                    return True

            except Exception as e:
                # Continue listening even if one chunk fails
                continue

        # Timeout reached
        console.print("[yellow]⚠ Wake word listening timed out[/yellow]")
        return False

    except KeyboardInterrupt:
        console.print("\n[yellow]Wake word listening cancelled[/yellow]")
        return False


def _play_beep(frequency=440, duration=0.1):
    """Play a simple beep sound."""
    try:
        import winsound
        winsound.Beep(int(frequency), int(duration * 1000))
    except Exception:
        pass  # Ignore beep errors


# ===============================
# Speech-to-Text (STT)
# ===============================
def record_audio(duration=None, samplerate=None):
    """Record audio from microphone."""
    duration = duration or CONFIG["recording_duration"]
    samplerate = samplerate or CONFIG["samplerate"]

    console.print(f"[cyan]🎤 Recording for {duration} seconds...[/cyan]")
    console.print("[dim]Speak now! Press Ctrl+C to stop early.[/dim]")

    try:
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        return np.squeeze(audio)
    except Exception as e:
        console.print(f"[red]Recording error: {e}[/red]")
        return None

def transcribe_audio(audio, samplerate=None, language=None):
    """Transcribe audio using Whisper."""
    if audio is None:
        return ""

    samplerate = samplerate or CONFIG["samplerate"]
    language = language or state["current_language"]
    stt = init_stt_model()

    with show_loading("Transcribing speech..."):
        try:
            segments, _ = stt.transcribe(
                audio,
                beam_size=5,
                language=language,
                vad_filter=True
            )
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
# LLM Integration with Streaming & Memory
# ===============================
def select_model():
    """Let user select an Ollama model or Claude model."""
    models = get_available_models()

    # Prepare the list of models including Claude options if available
    all_models = []

    # Add Claude models if the library is available
    if _ANTHROPIC_AVAILABLE:
        all_models.extend([
            "claude-sonnet-4",  # Claude Sonnet 4 Opus
            "claude-sonnet-3.5",  # Claude Sonnet 3.5
            "claude-haiku-3",     # Claude Haiku 3
        ])

    # Add Ollama models
    all_models.extend(models)

    if not all_models:
        console.print("[yellow]⚠ No models found (local or Claude)[/yellow]")
        console.print("[dim]Attempting to use default model...[/dim]")
        return CONFIG["default_model"]

    clear_screen()
    panel = Panel(
        "[bold cyan]Select a Model[/bold cyan]\n\n"
        + "\n".join(f"[bold]{i}.[/bold] {m}" for i, m in enumerate(all_models, 1)),
        title="Model Selection",
        border_style="blue"
    )
    console.print(panel)

    choices = [str(i) for i in range(1, len(all_models) + 1)]
    choice = Prompt.ask("Enter model number", choices=choices, default="1")

    selected = all_models[int(choice) - 1]

    # If it's a Claude model, ask for API key
    if selected.startswith("claude-"):
        if _ANTHROPIC_AVAILABLE:
            api_key = state["config"].api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                console.print("\n[bold]Claude API Key Required[/bold]")
                api_key = Prompt.ask("Enter your Anthropic API key", password=True)

            if api_key:
                state["config"].api_key = api_key
                save_config(state["config"])
                console.print(f"[green]✓[/green] API key saved in config")
            else:
                console.print("[red]⚠ No API key provided. Claude models won't work.[/red]")
        else:
            console.print(f"[red]⚠ Anthropic library not available. Cannot use {selected}[/red]")

    console.print(f"[green]✓[/green] Selected model: [bold]{selected}[/bold]")
    return selected

def select_language():
    """Let user select output language."""
    clear_screen()
    panel = Panel(
        "[bold cyan]Select Language[/bold cyan]\n\n"
        + "\n".join(f"[bold]{i}.[/bold] {LANGUAGES[code]['name']} ({code})"
                    for i, code in enumerate(LANGUAGES.keys(), 1)),
        title="Language Selection",
        border_style="green"
    )
    console.print(panel)

    choices = [str(i) for i in range(1, len(LANGUAGES) + 1)]
    choice = Prompt.ask("Enter language number", choices=choices, default="1")

    lang_code = list(LANGUAGES.keys())[int(choice) - 1]
    state["current_language"] = lang_code
    # Reset TTS model to reload with new language
    state["tts_model"] = None
    console.print(f"[green]✓[/green] Selected language: [bold]{LANGUAGES[lang_code]['name']}[/bold]")
    return lang_code


def select_wake_word():
    """Let user select or enter a custom wake word."""
    clear_screen()
    panel = Panel(
        "[bold cyan]Wake Word Setup[/bold cyan]\n\n"
        "The wake word activates voice mode.\n"
        "Say the wake word, then speak your query.\n\n"
        "[bold]Options:[/bold]\n"
        "1. Use default: 'Ustaad'\n"
        "2. Use default: 'Hey Assistant'\n"
        "3. Use default: 'Computer'\n"
        "4. Enter custom wake word\n"
        "5. Disable wake word",
        title="Wake Word Configuration",
        border_style="green"
    )
    console.print(panel)

    choices = ["1", "2", "3", "4", "5"]
    choice = Prompt.ask("Enter choice", choices=choices, default="1")

    if choice == "1":
        wake_word = "ustaad"
    elif choice == "2":
        wake_word = "hey assistant"
    elif choice == "3":
        wake_word = "computer"
    elif choice == "4":
        custom_panel = Panel(
            "[bold cyan]Enter Custom Wake Word[/bold cyan]\n\n"
            "Enter a word or short phrase (2-4 syllables works best).\n"
            "Examples: 'Jarvis', 'Alexa', 'Hey Siri', 'Hello'",
            title="Custom Wake Word",
            border_style="blue"
        )
        console.print(custom_panel)
        wake_word = Prompt.ask("Enter wake word").strip().lower()
        if not wake_word:
            wake_word = "ustaad"
    else:
        wake_word = ""

    state["config"].wake_word = wake_word
    state["config"].wake_word_enabled = (choice != "5")

    if choice == "5":
        console.print("[yellow]⚠ Wake word disabled[/yellow]")
    else:
        display_word = wake_word or "(none)"
        console.print(f"[green]✓[/green] Wake word: [bold]{display_word}[/bold]")
        if choice == "4":
            console.print("[dim]Tip: Use 2-3 syllable words for best detection[/dim]")

    return wake_word


def add_to_history(role: str, content: str):
    """Add a message to conversation history."""
    if not state["config"].history_enabled:
        return

    state["conversation_history"].append(
        ConversationMessage(role=role, content=content)
    )

    # Trim history to max length
    max_msgs = state["config"].max_history or CONFIG["max_history_messages"]
    if len(state["conversation_history"]) > max_msgs:
        state["conversation_history"] = state["conversation_history"][-max_msgs:]

def clear_history():
    """Clear conversation history."""
    state["conversation_history"] = []
    console.print("[green]✓[/green] Conversation history cleared")

def get_history_messages() -> List[Dict]:
    """Get conversation history as list of dicts for LLM."""
    return [
        {"role": msg.role, "content": msg.content}
        for msg in state["conversation_history"]
    ]

def export_history(filename: str | None = None):
    """Export conversation history to a file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(msg) for msg in state["conversation_history"]],
                f,
                indent=2,
                ensure_ascii=False
            )
        console.print(f"[green]✓[/green] History exported to: {filename}")
    except Exception as e:
        console.print(f"[red]Export error: {e}[/red]")

def get_llm_response(prompt: str, stream: bool = True) -> str:
    """Get response from Ollama LLM with streaming and conversation memory."""
    model = state["selected_model"]
    if not model:
        return "Error: No model selected."

    if not check_ollama_connection():
        return "Error: Ollama is not running. Start Ollama first: 'ollama serve'"

    full_response = []
    console.print(f"[cyan]🤖 Thinking with [bold]{model}[/bold]...[/cyan]")

    try:
        # Build messages with conversation history
        messages = get_history_messages()

        # Add system prompt for language consistency
        lang_name = LANGUAGES.get(state["current_language"], {}).get("name", "English")
        system_msg = f"You are a helpful voice assistant. Respond in {lang_name}."
        messages.insert(0, {"role": "system", "content": system_msg})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        if stream:
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=True
            )

            with console.status("[bold cyan]Generating response...", spinner="dots") as status:
                for chunk in response:
                    if chunk.get("done") is True:
                        break
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_response.append(content)
                        preview = "".join(full_response)[-50:] if len("".join(full_response)) > 50 else "".join(full_response)
                        status.update(f"[bold cyan]Generating:[/bold cyan] {preview}...")

            full_text = "".join(full_response)
        else:
            response = ollama.chat(
                model=model,  # type: ignore
                messages=messages,
                timeout=CONFIG["ollama_timeout"]
            )
            full_text = response.get("message", {}).get("content", "")

        result = full_text.strip() if full_text else "I couldn't generate a response."
        return result

    except ollama.ResponseError as e:
        error_msg = f"Ollama error: {e.error}"
        if "model not found" in error_msg.lower():
            error_msg += "\nTry running: ollama pull " + model
        return error_msg
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

# ===============================
# Text-to-Speech (TTS)
# ===============================
import re

# Regex pattern to match emojis and many Unicode symbols
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002600-\U000026FF"  # Misc symbols
    "]+",
    flags=re.UNICODE
)


def clean_text_for_tts(text: str) -> str:
    """Remove emojis and unsupported characters for TTS."""
    if not text:
        return ""

    # Replace emojis with placeholder text
    text = _EMOJI_PATTERN.sub(lambda m: " smile " if "😊" in m.group() else "", text)

    # Remove any remaining non-ASCII characters (keep basic punctuation)
    cleaned = ""
    for char in text:
        if ord(char) < 128 or char in ".,!?;:()[]{}'\"\n- ":
            cleaned += char
        elif char in "àèéìòù":
            cleaned += char  # Keep accented chars for non-English
        else:
            cleaned += " "  # Replace unknown chars with space

    # Clean up multiple spaces
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def speak(text: str):
    """Convert text to speech and play it."""
    if not text or not state["config"].tts_enabled:
        return

    # Clean text for TTS
    original_text = text
    text = clean_text_for_tts(text)

    # Check if text is too short after cleaning
    if len(text) < 5:
        console.print("[yellow]⚠ Text too short for speech generation[/yellow]")
        return

    output_file = "response.wav"

    # Truncate very long text for TTS
    max_chars = 500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
        console.print("[yellow]⚠ Response truncated for speech[/yellow]")

    console.print(f"[yellow]🔊 Speaking...[/yellow]")

    # Try gTTS for non-English or as fallback
    if state["current_language"] != "en" and _GTTS_AVAILABLE:
        _speak_gtts(text, output_file, original_text)
        return

    # Try Coqui TTS for English
    tts = init_tts_model()
    try:
        with console.status("[bold yellow]Generating speech...", spinner="dots"):
            tts.tts_to_file(text=text, file_path=output_file)
        _play_audio(output_file)
    except Exception as e:
        error_msg = str(e).lower()
        if "kernel size" in error_msg or "padded input" in error_msg:
            console.print("[yellow]⚠ Text too short for TTS model[/yellow]")
        elif "vocabulary" in error_msg or "not found" in error_msg:
            console.print("[yellow]⚠ Unsupported characters in text[/yellow]")
        elif _GTTS_AVAILABLE:
            console.print("[yellow]⚠ Trying Google TTS as fallback...[/yellow]")
            _speak_gtts(text, output_file, original_text)
        else:
            console.print(f"[red]⚠ TTS Error: {e}[/red]")
            console.print(f"[dim]Original:[/dim] {original_text[:100]}")
            console.print(f"[dim]Cleaned:[/dim] {text[:100]}")


def _speak_gtts(text: str, output_file: str, original_text: str):
    """Use gTTS for text-to-speech (requires internet)."""
    try:
        lang = LANGUAGES.get(state["current_language"], {}).get("code", "en")
        with console.status("[bold yellow]Generating speech (Google TTS)...", spinner="dots"):
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(output_file)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]⚠ gTTS Error: {e}[/red]")
        console.print("[dim]Response:[/dim]")
        console.print(Panel(original_text, border_style="yellow"))


def _play_audio(output_file: str):
    """Play audio file cross-platform."""
    import shlex

    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(output_file, winsound.SND_FILENAME)
        except ImportError:
            # Using subprocess is safer than os.system
            subprocess.run(['cmd', '/c', 'start', '/B', '/WAIT', '', output_file], shell=False)
    elif sys.platform == "darwin":
        subprocess.run(['afplay', output_file])
    else:
        # For Linux, use subprocess with proper argument separation
        subprocess.run(['aplay', output_file])

# ===============================
# UI Functions
# ===============================
def display_banner():
    """Display the main banner."""
    clear_screen()
    config_obj = state.get("config", AppConfig())
    wake_word = getattr(config_obj, 'wake_word', 'Ustaad') or "Ustaad"
    banner = Panel(
        Text(
            "🎙️ VOICE ASSISTANT v2.0\n\n"
            "[bold]Commands:[/bold]\n"
            "[M] Change mode    [L] Change language\n"
            "[W] Wake toggle    [H] TTS toggle\n"
            "[E] Export chat    [C] Clear history\n"
            "[Q] Quit           [ENTER] Continue\n\n"
            f"[dim]Voice mode: Say '{wake_word}' to activate[/dim]",
            justify="center",
            style="bold cyan"
        ),
        title="Welcome",
        border_style="blue",
        expand=False
    )
    console.print(banner)
    console.print()

def display_mode_selection():
    """Show mode selection menu."""
    clear_screen()
    panel = Panel(
        "[bold cyan]Select Input Mode[/bold cyan]\n\n"
        "[bold]1.[/bold] 💬 Text Mode - Type your queries\n"
        "[bold]2.[/bold] 🎤 Voice Mode - Speak your queries",
        title="Mode Selection",
        border_style="green"
    )
    console.print(panel)

    choice = Prompt.ask("Enter your choice", choices=["1", "2"], default="1")
    return "text" if choice == "1" else "voice"

def display_status():
    """Display current configuration status."""
    lang_info = LANGUAGES.get(state["current_language"], {})
    history_count = len(state["conversation_history"])

    wake_status = "[green]ON[/green]" if state["config"].wake_word_enabled else "[red]OFF[/red]"
    wake_word = state["config"].wake_word or "Ustaad"

    status_items = [
        f"Mode: [bold]{state['current_mode'].upper()}[/bold]",
        f"Model: [bold]{state['selected_model']}[/bold]",
        f"Language: [bold]{lang_info.get('name', 'Unknown')}[/bold]",
        f"TTS: {'[green]ON[/green]' if state['config'].tts_enabled else '[red]OFF[/red]'}",
        f"History: [cyan]{history_count}[/cyan] msgs",
        f"Wake: {wake_status} ('{wake_word}')",
        f"STT: {'[green]✓[/green]' if state["stt_model"] else '[yellow]...[/yellow]'}",
    ]

    status_panel = Panel(
        " | ".join(status_items),
        title="Status",
        border_style="dim"
    )
    console.print(status_panel)

def show_help():
    """Show help message."""
    config_obj = state.get("config", AppConfig())
    wake_word = getattr(config_obj, 'wake_word', 'Ustaad') or "Ustaad"
    help_text = f"""
    [bold cyan]Keyboard Commands:[/bold cyan]
    • [M] Change input mode (text/voice)
    • [L] Change output language
    • [W] Toggle wake word detection
    • [H] Toggle TTS on/off
    • [E] Export conversation history
    • [C] Clear conversation history
    • [Q] Quit the application
    • [ENTER] Continue to next input

    [bold cyan]Voice Mode:[/bold cyan]
    • Say [italic]'{wake_word}'[/italic] to activate (when wake word is enabled)
    • Then speak your query normally

    [bold cyan]Tips:[/bold cyan]
    • Say [italic]'clear'[/italic] to clear history
    • Say [italic]'help'[/italic] to get assistance
    • Conversation context is maintained for better responses
    """
    console.print(Panel(help_text, title="Help", border_style="blue"))

# ===============================
# Main Interaction Loop
# ===============================
def handle_text_input() -> str:
    """Handle text input from user."""
    return Prompt.ask("\n[bold green]You:[/bold green] ").strip()

def handle_voice_input() -> str:
    """Handle voice input with wake word detection."""
    # Check if wake word detection is enabled
    if state["config"].wake_word_enabled:
        wake_detected = listen_for_wake_word()

        if not wake_detected:
            console.print("[yellow]⚠ Wake word not detected. Try again.[/yellow]")
            return ""

    # Record the actual query
    audio = record_audio()
    return transcribe_audio(audio)

def process_query(prompt: str) -> bool | str:
    """Process a user query and generate response."""
    if not prompt:
        console.print("[yellow]⚠ No input detected. Please try again.[/yellow]")
        return False

    # Check for commands
    cmd = prompt.lower().strip()
    if cmd in ["q", "quit", "exit"]:
        return "quit"
    if cmd == "m":
        return "mode"
    if cmd == "l":
        return "language"
    if cmd == "w":
        state["config"].wake_word_enabled = not state["config"].wake_word_enabled
        status = "enabled" if state["config"].wake_word_enabled else "disabled"
        wake_name = "Ustaad" if state["config"].wake_word_enabled else ""
        console.print(f"[green]✓[/green] Wake word {status} {wake_name}")
        save_config(state["config"])
        return False
    if cmd == "h":
        state["config"].tts_enabled = not state["config"].tts_enabled
        status = "enabled" if state["config"].tts_enabled else "disabled"
        console.print(f"[green]✓[/green] TTS {status}")
        save_config(state["config"])
        return False
    if cmd == "e":
        export_history()
        return False
    if cmd == "c":
        clear_history()
        return False
    if cmd in ["help", "?"]:
        show_help()
        return False

    # Show user input
    console.print(Panel(
        f"[bold]Your query:[/bold] {prompt}",
        border_style="green",
        expand=False
    ))

    # Get LLM response
    response = get_llm_response(prompt)

    # Display response
    response_panel = Panel(
        f"[bold]Assistant:[/bold]\n{response}",
        border_style="magenta",
        expand=False
    )
    console.print(response_panel)

    # Add to history
    add_to_history("user", prompt)
    add_to_history("assistant", response)

    # Speak the response
    speak(response)

    return True

def main():
    """Main application loop."""
    # Load saved config
    state["config"] = load_config()

    # Welcome message
    display_banner()
    console.print("[dim]Initializing...[/dim]\n")

    # Check Ollama
    if check_ollama_connection():
        console.print("[green]✓ Ollama is running[/green]")
    else:
        console.print("[yellow]⚠ Ollama not detected. It will be checked when you send a query.[/yellow]")

    # Select model (always show selection menu)
    state["selected_model"] = select_model()
    state["config"].model = state["selected_model"]
    save_config(state["config"])

    # Select language (always show selection menu)
    state["current_language"] = select_language()
    state["config"].language = state["current_language"]
    save_config(state["config"])

    # Select input mode (always show selection menu)
    state["current_mode"] = display_mode_selection()
    state["config"].input_mode = state["current_mode"]
    save_config(state["config"])

    # Select wake word (always show selection menu)
    wake_word = select_wake_word()
    state["config"].wake_word = wake_word
    state["config"].wake_word_enabled = bool(wake_word)
    save_config(state["config"])

    # Main loop
    while state["running"]:
        try:
            display_banner()
            display_status()
            console.print()

            # Get input based on mode
            if state["current_mode"] == "text":
                prompt = handle_text_input()
            else:
                prompt = handle_voice_input()

            # Process the query
            result = process_query(prompt)

            if result == "quit":
                console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
                break
            elif result == "mode":
                state["current_mode"] = display_mode_selection()
                state["config"].input_mode = state["current_mode"]
                save_config(state["config"])
            elif result == "language":
                state["current_language"] = select_language()
                state["config"].language = state["current_language"]
                save_config(state["config"])
            elif result:
                # Successful query, continue
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]Goodbye! 👋[/bold cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]⚠ Error: {e}[/red]")
            console.print("[dim]Press Enter to continue...[/dim]")
            try:
                input()
            except KeyboardInterrupt:
                console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
                break

if __name__ == "__main__":
    main()
