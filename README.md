# Multilingual Voice Assistant

A powerful voice assistant that supports multiple languages (English, Urdu, and Pashto) with speech-to-text, natural language processing, and text-to-speech capabilities.

## Developer
**M. Fawad Baig**

## Features

- 🎤 Voice and text input support
- 🌐 Multilingual support (English, Urdu, Pashto)
- 🗣️ Speech-to-text using Whisper
- 🤖 Natural language processing using Ollama
- 🔊 Text-to-speech synthesis
- 💻 Interactive command-line interface
- 🎯 Real-time audio processing
- 🔄 Automatic language detection and translation

## Dependencies

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Audio input device (microphone)
- Audio output device (speakers)

### System Packages
```bash
# For Fedora
sudo dnf install mplayer ffmpeg

# For Ubuntu/Debian
sudo apt-get install mplayer ffmpeg
```

### Python Packages
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/mfbgull/voice_assistant.git
cd voice_assistant
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv va-env
source va-env/bin/activate  # Linux/Mac
# or
.\va-env\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Start the voice assistant:
```bash
python va2.py
```

2. Select your preferred input mode:
   - Text Mode (1): Type your queries
   - Voice Mode (2): Speak your queries

3. Choose your preferred language:
   - ur: Urdu
   - en: English
   - ps: Pashto

4. Interact with the assistant:
   - In text mode: Type your query and press Enter
   - In voice mode: Speak after the countdown
   - Press 'm' to change input mode
   - Press Ctrl+C to exit

## Architecture

The project consists of several key components:

1. **Speech-to-Text (STT)**
   - Uses Whisper model for accurate speech recognition
   - Supports multiple languages
   - Real-time audio processing

2. **Language Processing**
   - Integrates with Ollama for natural language understanding
   - Provides intelligent responses
   - Handles context and conversation flow

3. **Translation**
   - Uses Google Translator for language conversion
   - Supports translation between English, Urdu, and Pashto
   - Handles edge cases and ensures reliable output

4. **Text-to-Speech (TTS)**
   - Uses gTTS for high-quality speech synthesis
   - Supports multiple languages
   - Natural-sounding voice output

## Project Structure

```
voice_assistant/
├── va2.py              # Main application file
├── README.md           # Documentation
└── requirements.txt    # Python dependencies
```

## Known Issues

- Pashto TTS falls back to Urdu due to limited language support
- Some audio devices may require additional configuration
- Internet connection required for translation services

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Ollama for natural language processing
- Google Translate for translation services
- gTTS for text-to-speech conversion
