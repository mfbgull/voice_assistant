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
import importlib.metadata

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

# Try to import Supertonic (lightning-fast on-device multilingual TTS)
try:
    from supertonic import TTS as SupertonicTTS
    _SUPER_TONIC_AVAILABLE = True
except ImportError:
    _SUPER_TONIC_AVAILABLE = False
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

# Try to import Coqui TTS (may not be available on Python 3.12+)
try:
    from TTS.api import TTS
    _COQUI_TTS_AVAILABLE = True
except ImportError:
    _COQUI_TTS_AVAILABLE = False
    TTS = None

# Try to import sanotts (sanoTTS) - numpy-only TTS
try:
    import sanotts
    _SANOTTS_AVAILABLE = True
except ImportError:
    _SANOTTS_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
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

# Supertonic 3 supported language codes (verified at runtime)
SUPERTONIC_LANGUAGES = {"en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el",
                        "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it",
                        "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl",
                        "sv", "tr", "uk", "vi"}

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
    tts_engine: str = "supertonic"  # 'whisper' (Coqui), 'kitten', 'supertonic', 'gtts', or 'sanotts'
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
    "supertonic_tts": None,
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

def init_supertonic_tts():
    """Initialize Supertonic TTS model with loading indicator."""
    if state["supertonic_tts"] is None:
        if not _SUPER_TONIC_AVAILABLE:
            console.print("[red]Error: Supertonic library not found. Install with: pip install supertonic[/red]")
            return None
        with show_loading("Loading Supertonic TTS (99M multilingual model)..."):
            try:
                state["supertonic_tts"] = SupertonicTTS(auto_download=True)
            except Exception as e:
                console.print(f"[red]Error initializing Supertonic TTS: {e}[/red]")
                return None
    return state["supertonic_tts"]

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
    
    panel_text = """[bold cyan]Select TTS Engine[/bold cyan]

[bold]1.[/bold] Supertonic (99M multilingual) - [green]Recommended[/green] Fast, 31 languages, on-device
[bold]2.[/bold] KittenTTS (80M) - Fast, realistic, CPU optimized"""
    
    if _COQUI_TTS_AVAILABLE:
        panel_text += """
[bold]3.[/bold] Whisper (Coqui TTS) - High quality, slower"""
        panel_text += """
[bold]4.[/bold] Google gTTS - Clear but requires internet"""
        default_choice = "1"
        choices_list = ["1", "2", "3", "4"]
    else:
        panel_text += """
[bold]3.[/bold] Google gTTS - Clear but requires internet"""
        panel_text += """
[dim]Note: Coqui TTS not available on Python 3.12+[/dim]"""
        default_choice = "1"
        choices_list = ["1", "2", "3"]
    
    if not _SUPER_TONIC_AVAILABLE:
        panel_text = panel_text.replace("[bold]1.[/bold] Supertonic (99M multilingual) - [green]Recommended[/green] Fast, 31 languages, on-device",
                                         "[bold]1.[/bold] Supertonic [dim](not installed - pip install supertonic)[/dim]")
    
    # Add sanotts option if available
    if _SANOTTS_AVAILABLE:
        panel_text += """
[bold]5.[/bold] sanotts (sanoTTS) - Lightweight, numpy-only, ~1.4M params"""
        choices_list = ["1", "2", "3", "4", "5"]
    else:
        choices_list = ["1", "2", "3", "4"]
    
    panel = Panel(panel_text, title="TTS Selection", border_style="magenta")
    console.print(panel)
    choice = Prompt.ask("Enter choice", choices=choices_list, default=default_choice)
    
    if choice == "1":
        engine = "supertonic"
    elif choice == "2":
        engine = "kitten"
    elif choice == "3":
        engine = "whisper" if _COQUI_TTS_AVAILABLE else "gtts"
    elif choice == "4":
        engine = "gtts"
    elif _SANOTTS_AVAILABLE and choice == "5":
        engine = "sanotts"
    else:
        engine = "gtts"
    
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
    
    # Supertonic supports multilingual text natively (including non-ASCII),
    # so skip ASCII-only cleaning for it
    if state["config"].tts_engine == "supertonic":
        cleaned_text = " ".join(text.split()).strip()
    else:
        cleaned_text = clean_text_for_tts(text)
    
    if len(cleaned_text) < 2: return

    output_file = "response.wav"
    console.print(f"[yellow]🔊 Speaking ({state['config'].tts_engine})...[/yellow]")

    if state["config"].tts_engine == "supertonic" and _SUPER_TONIC_AVAILABLE:
        _speak_supertonic(cleaned_text, output_file)
    elif state["config"].tts_engine == "kitten" and _KITTENTTS_AVAILABLE:
        _speak_kitten(cleaned_text, output_file)
    elif state["config"].tts_engine == "gtts" and _GTTS_AVAILABLE:
        _speak_gtts(cleaned_text, output_file)
    elif state["config"].tts_engine == "sanotts" and _SANOTTS_AVAILABLE:
        _speak_sano(cleaned_text, output_file)
    else:
        _speak_whisper(cleaned_text, output_file)

def _speak_whisper(text, output_file):
    """Coqui TTS (Whisper labeled)."""
    if not _COQUI_TTS_AVAILABLE:
        console.print("[dim]⚠ Coqui TTS not available (Python 3.12+ incompatibility)[/dim]")
        if _GTTS_AVAILABLE:
            _speak_gtts(text, output_file)
        return
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

def _speak_sano(text, output_file):
    """sanoTTS (sanoTTS) - Lightweight, numpy-only TTS engine.
    
    Uses the sanotts Synthesizer with default voices.
    Voices available: amy, kristin, hfc, id, vi (piperlite) and nano voices.
    """
    if not _SANOTTS_AVAILABLE:
        console.print("[dim]⚠ sanotts library not available[/dim]")
        return
    
    try:
        with console.status("[bold yellow]Generating sanotts TTS...", spinner="dots"):
            # Use a default piperlite voice (amy is commonly available)
            synth = sanotts.Synthesizer(voice="amy")
            result = synth.synthesize(text)
            
            # audio is already a numpy float32 array in [-1, 1]
            audio = result.audio
            
            # Ensure audio is saved at good quality
            sf.write(output_file, audio, result.sample_rate)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]sanotts TTS Error: {e}[/red]")
        # Fall back to whisper/Coqui TTS
        _speak_whisper(text, output_file)

def _speak_kitten(text, output_file):
    """KittenTTS."""
    kitten = init_kitten_tts()
    if not kitten:
        _speak_whisper(text, output_file)
        return
    try:
        with console.status("[bold yellow]Generating KittenTTS...", spinner="dots"):
            # Try different voices for better quality
            # Available voices: 'voice-1', 'voice-2', 'expr-voice-1', 'expr-voice-2', 'expr-voice-2-m'
            # 'voice-1' and 'voice-2' are cleaner, 'expr-*' are more expressive but may have more noise
            audio = kitten.generate(text, voice='voice-1')
            
            # Normalize audio to reduce noise
            import numpy as np
            audio = np.array(audio, dtype=np.float32)
            
            # Apply simple normalization
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio * 0.75 / max_amp  # Normalize to 75% to prevent clipping
            
            # Apply gentle low-pass filter to reduce high-frequency noise
            # Simple moving average filter
            window_size = 3
            if len(audio) > window_size:
                audio = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
            
            # Save at higher quality (24kHz is good for speech)
            sf.write(output_file, audio, 24000)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]KittenTTS Error: {e}[/red]")
        _speak_whisper(text, output_file)

def _speak_supertonic(text, output_file):
    """Supertonic TTS - fast on-device multilingual."""
    tts = init_supertonic_tts()
    if not tts:
        _speak_gtts(text, output_file)
        return
    
    # Map current language to Supertonic language code
    lang_code = state["current_language"]
    if lang_code not in SUPERTONIC_LANGUAGES:
        # Use language-agnostic mode for unsupported languages (ur, ps, etc.)
        lang_code = "na"
    
    try:
        with console.status("[bold yellow]Generating Supertonic TTS...", spinner="dots"):
            style = tts.get_voice_style(voice_name="M1")
            wav, duration = tts.synthesize(
                text=text,
                lang=lang_code,
                voice_style=style,
                total_steps=8,
                speed=1.05
            )
            tts.save_audio(wav, output_file)
        _play_audio(output_file)
    except Exception as e:
        console.print(f"[red]Supertonic TTS Error: {e}[/red]")
        # Fallback to gTTS
        if _GTTS_AVAILABLE:
            console.print("[yellow]Falling back to gTTS...[/yellow]")
            _speak_gtts(text, output_file)

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
# Engine & System Info Display
# ===============================
def get_engine_display_info() -> Dict[str, str]:
    """Return engine-specific display info (name, model, features)."""
    engine = state["config"].tts_engine
    
    if engine == "supertonic" and _SUPER_TONIC_AVAILABLE:
        try:
            st_ver = importlib.metadata.version('supertonic')
        except Exception:
            st_ver = "?"
        return {
            "name": f"Supertonic v{st_ver}",
            "model": "99M multilingual ONNX",
            "langs": "31 languages",
            "quality": "44.1kHz · studio-grade",
            "runtime": "ONNX · on-device",
        }
    elif engine == "kitten":
        return {
            "name": "KittenTTS",
            "model": "KittenML/kitten-tts-mini-0.8",
            "langs": "English only",
            "quality": "24kHz · CPU optimized",
            "runtime": "ONNX · on-device",
        }
    elif engine == "whisper":
        return {
            "name": "Coqui TTS",
            "model": "tacotron2-DDC",
            "langs": "English only",
            "quality": "Standard",
            "runtime": "PyTorch",
        }
    else:  # gtts
        return {
            "name": "Google gTTS",
            "model": "Cloud TTS",
            "langs": "Multilingual",
            "quality": "Standard · requires internet",
            "runtime": "Cloud API",
        }

def get_system_status() -> Dict[str, str]:
    """Return current system status indicators."""
    status = {}
    
    # LLM model
    status["llm"] = state["selected_model"] or "None"
    
    # Wake word
    ww = state["config"].wake_word
    if ww and state["config"].wake_word_enabled:
        status["wake_word"] = f'"{ww}" 🟢'
    elif ww and not state["config"].wake_word_enabled:
        status["wake_word"] = "Disabled"
    else:
        status["wake_word"] = "None"
    
    # History
    hcount = len(state["conversation_history"]) // 2
    status["history"] = f"{hcount} exchanges"
    
    # Ollama status
    ollama_ok = check_ollama_connection()
    status["ollama"] = "🟢 Connected" if ollama_ok else "🔴 Disconnected"
    
    return status

def display_banner():
    """Display the main UI banner with engine and system info."""
    clear_screen()
    config = state["config"]
    
    # Gather display data
    engine_info = get_engine_display_info()
    sys_status = get_system_status()
    
    lang_name = LANGUAGES.get(state["current_language"], {}).get("name", "Unknown")
    lang_code = state["current_language"]
    
    # Engine features bar
    features = engine_info.get("quality", "") + "  ·  " + engine_info.get("runtime", "")
    if engine_info.get("langs"):
        features = engine_info["langs"] + "  ·  " + features
    
    # Build banner content
    banner_content = (
        f"[bold cyan]🎙️  V O I C E   A S S I S T A N T   v2.1[/bold cyan]\n"
        f"[dim]─[/dim]" * 48 + "\n\n"
        f"[bold]TTS Engine:[/bold]    [green]{engine_info['name']}[/green]\n"
        f"[bold]Model:[/bold]          [yellow]{engine_info['model']}[/yellow]\n"
        f"[bold]Features:[/bold]       [dim]{features}[/dim]\n"
        f"\n"
        f"[bold]Language:[/bold]       {lang_name} [dim]({lang_code})[/dim]\n"
        f"[bold]Input Mode:[/bold]     {'🎤 Voice' if state['current_mode'] == 'voice' else '⌨️ Text'}\n"
        f"[bold]LLM Model:[/bold]      [magenta]{sys_status['llm']}[/magenta]\n"
        f"\n"
        f"[bold]Wake Word:[/bold]      {sys_status['wake_word']}\n"
        f"[bold]History:[/bold]        {sys_status['history']}\n"
        f"[bold]Ollama:[/bold]         {sys_status['ollama']}\n"
    )
    
    # Adapt panel width to terminal size, cap at 54
    term_width = shutil.get_terminal_size().columns
    panel_width = min(term_width - 2, 54)
    
    banner = Panel(
        banner_content,
        title="[bold blue]Dashboard[/bold blue]",
        border_style="blue",
        padding=(1, 2),
        width=panel_width
    )
    console.print(banner)

def select_input_mode():
    """Let user select input mode."""
    clear_screen()
    panel = Panel(
        """[bold cyan]Select Input Mode[/bold cyan]

[bold]1.[/bold] Text Input - Type your queries
[bold]2.[/bold] Voice Input - Speak your queries (requires microphone)""",
        title="Input Mode Selection",
        border_style="cyan"
    )
    console.print(panel)
    choice = Prompt.ask("Enter mode number", choices=["1", "2"], default="2")
    mode = "text" if choice == "1" else "voice"
    console.print(f"[bold cyan]Selected Input Mode:[/bold cyan] {mode.upper()}")
    return mode

def main():
    state["config"] = load_config()
    display_banner()

    # Selection Flow
    state["selected_model"] = select_model()
    state["current_language"] = select_language()
    state["config"].tts_engine = select_tts_engine()
    state["current_mode"] = select_input_mode()
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
