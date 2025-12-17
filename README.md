# 🤖 JARVIS - Discord Voice AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Discord](https://img.shields.io/badge/discord.py-2.0+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*Advanced Discord bot with real-time voice transcription, AI responses, and comprehensive analytics*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Premium](#-premium-version)

</div>

---

## ✨ Features

### 🎙️ Voice Recognition
- **Real-time transcription** using OpenAI Whisper (distil-large-v3)
- **Emotion detection** from voice tone and pitch
- **Voice biometrics** - unique profile for each user
- **Multi-language support**

### 📊 Analytics & Stats
- **Leaderboards**: Who talks most, top words, swear word tracking
- **Conversation analysis**: Topics, relationships, speaking patterns
- **Time-based filters**: Today, yesterday, this week stats
- **Activity logging**: Full message and voice history

### 🎵 Music Integration
- **Spotify integration** with playlist support
- **YouTube playback**
- **Auto DJ** mode (plays music based on conversation energy)
- **Queue management** and soundboard

### 🤖 AI Features
- **Ollama integration** for intelligent responses
- **Web search** capability for current information
- **Context-aware** responses using conversation history
- **200+ voice commands**

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU (for Whisper transcription)
- Ollama installed
- Discord Bot Token

### Quick Start
```bash
# Clone repository
git clone https://github.com/[your-username]/jarvis-lite
cd jarvis-lite

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Discord token and API keys

# Run bot
python bot_lite.py YOUR_DISCORD_TOKEN
```

### Configuration

Create a `.env` file:
```env
DISCORD_TOKEN=your_bot_token_here
SPOTIPY_CLIENT_ID=your_spotify_id (optional)
SPOTIPY_CLIENT_SECRET=your_spotify_secret (optional)
```

---

## 📝 Usage

### Voice Commands

**Stats & Analytics:**
- `Jarvis, who talks the most?`
- `Jarvis, top swear words`
- `Jarvis, my stats`
- `Jarvis, who talked the most yesterday?`

**Music:**
- `Jarvis, play [song name]`
- `Jarvis, skip`
- `Jarvis, enable auto DJ`

**General:**
- `Jarvis, [any question]` - AI-powered responses
- `Jarvis, what's the weather in [city]?`
- `Jarvis, search for [topic]`

**Full command list:** [See documentation](docs/COMMANDS.md)

---

## 💎 Premium Version

**Want advanced psychological warfare features?**

The premium version includes:
- 🔥 **17 Psychological Warfare Features**
- 🎯 Behavioral pattern analysis
- 📊 Loyalty tracking & social dynamics
- 💀 Relationship destroyer suite
- 🎭 Gaslighting & manipulation detection
- 🗣️ Shit-talk tracker
- 🧠 Advanced behavioral analytics
- ⚡ Priority support

**[Contact for Premium Access](https://github.com/CobCob047/jarvis-lite/issues)**

---

## 🛠️ Tech Stack

- **Discord.py** - Bot framework
- **Faster-Whisper** - Voice transcription
- **Ollama** - AI responses (Gemma2 27B)
- **Parselmouth** - Voice biometrics
- **DuckDuckGo** - Web search
- **SQLite** - Database

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## ⭐ Support

If you find this useful, please star the repository!

**Created by Cob / OneRaap Hosting**
- GitHub: [@CobCob047](https://github.com/CobCob047)

---

<div align="center">
  
**[⬆ back to top](#-jarvis---discord-voice-ai-assistant)**

</div>
