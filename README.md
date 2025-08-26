# 🎙️ Discord Meeting Transcription Bot (MeetMind)

A powerful Discord bot that provides real-time voice transcription, multi-language support, and AI-powered Q&A for your meeting channels. Perfect for keeping accurate records of voice conversations and making meeting content searchable and accessible.

## Link
[Click Here](https://meetmind.xyz/)
## ✨ Features

### 🔊 Real-Time Transcription
- Live voice-to-text transcription during Discord voice calls
- Automatic speaker identification
- Timestamp tracking for all messages
- Dedicated transcription channels for organized record-keeping

### 🌍 Multi-Language Support
- Support for 30+ languages including English, Spanish, French, German, Arabic, Chinese, Japanese, and more
- Automatic language detection
- High-accuracy speech recognition using OpenAI's Whisper model

### 📄 Document Generation
- Export transcripts in multiple formats: PDF, DOCX, and TXT
- Professional formatting with timestamps and speaker names
- Automatic file cleanup after delivery

### 🤖 AI-Powered Q&A
- Ask questions about meeting content using natural language
- Powered by OpenAI GPT for intelligent responses
- Works with both active and completed meetings
- Context-aware answers with speaker attribution

### 🛡️ Smart Meeting Management
- Auto-end meetings when all participants leave
- Meeting status monitoring and duration tracking
- Participant tracking and channel management
- Comprehensive error handling and logging

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Discord Bot Token
- OpenAI API Key (optional, for Q&A features)
- FFmpeg installed on your system

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/OmarAshry1/MeetMind.git
cd MeetMind
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```env
DISCORD_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the bot**
```bash
python MeetMind_local.py
```

## 📋 Commands

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `!start_meeting [language]` | Start a new meeting with transcription | `!start_meeting english`<br>`!start_meeting arabic`<br>`!start_meeting auto` |
| `!end_meeting [format]` | End meeting and generate transcript | `!end_meeting pdf`<br>`!end_meeting docx` |
| `!ask <question>` | Ask AI about meeting content | `!ask What did John say about the budget?` |
| `!meeting_status` | Check current meeting status | `!meeting_status` |
| `!languages` | Show supported languages | `!languages` |
| `!meeting_help` | Display help information | `!meeting_help` |

## 🌐 Supported Languages

The bot supports transcription in 30+ languages organized by region:

**🌍 European Languages**
- English, Spanish, French, German, Italian, Portuguese, Russian
- Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Czech
- Hungarian, Greek

**🏛️ Middle Eastern Languages**
- Arabic, Hebrew, Persian, Turkish, Urdu

**🏮 Asian Languages**
- Chinese (Mandarin), Japanese, Korean, Hindi
- Thai, Vietnamese, Indonesian, Malay, Tamil, Bengali

**🤖 Special Options**
- `auto` - Automatic language detection

## 📦 Dependencies

```txt
discord.py[voice]==2.3.2
discord-ext-voice-recv
python-dotenv==1.0.0
faster-whisper==0.10.0
python-docx==0.8.11
reportlab==4.0.7
openai==1.3.0
asyncio
```

## 🔧 Setup Instructions

### Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application and bot
3. Copy the bot token to your `.env` file
4. Enable the following bot permissions:
   - Read Messages
   - Send Messages
   - Connect (Voice)
   - Speak (Voice)
   - Use Voice Activity
   - Manage Channels (for creating transcription channels)

### OpenAI API Setup (Optional)

1. Sign up at [OpenAI](https://platform.openai.com/)
2. Generate an API key
3. Add it to your `.env` file
4. Note: The Q&A feature requires this API key

### Invite Bot to Server

Use this URL template (replace `YOUR_CLIENT_ID`):
```
https://discord.com/oauth2/authorize?client_id=1399826399472390234&permissions=3255344&integration_type=0&scope=bot+applications.commands
```

## 🖥️ Deployment

### Deploy on Render (Recommended)

1. **Create `requirements.txt`** (see Dependencies section above)

2. **Push to GitHub**

3. **Deploy on Render**
   - Create new "Background Worker" service
   - Connect your GitHub repository
   - Set environment variables (`DISCORD_TOKEN`, `OPENAI_API_KEY`)
   - Deploy!

For detailed deployment instructions, see our [Deployment Guide](docs/deployment.md).

### Alternative Deployment Options
- **Railway**: Great for Discord bots
- **Heroku**: Use Worker dyno type
- **DigitalOcean App Platform**: Worker components
- **Self-hosted**: VPS or dedicated server

## 💡 Usage Examples

### Starting a Meeting
```
User: !start_meeting arabic
Bot: ✅ Meeting started! Live transcriptions in Arabic will appear in #meeting-transcription-0824-1430
```

### During the Meeting
The bot automatically creates transcriptions like:
```
[14:32:15] **John**: I think we should increase the marketing budget for Q4
[14:32:28] **Sarah**: That's a good point, but we need to consider the ROI
[14:32:45] **Ahmed**: في رأيي، يجب أن نركز على الإعلانات الرقمية
```

### Asking Questions
```
User: !ask What did Ahmed say about digital advertising?
Bot: 🤖 Based on the meeting transcript, Ahmed said "في رأيي، يجب أن نركز على الإعلانات الرقمية" which translates to "In my opinion, we should focus on digital advertising" at timestamp [14:32:45].
```

## 📊 Performance Notes

- **Memory Usage**: ~500MB-1GB depending on concurrent meetings
- **CPU Usage**: Higher during active transcription (audio processing)
- **Latency**: ~2-7 seconds for transcription (configurable buffer)
- **Accuracy**: 85-95% depending on audio quality and language

## ⚠️ Limitations

- Requires stable internet connection for real-time transcription
- Audio quality affects transcription accuracy
- OpenAI API key required for Q&A features
- Voice channel participants must have clear microphones
- Bot cannot transcribe multiple simultaneous speakers perfectly

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/OmarAshry1/MeetMind.git
cd MeetMind
pip install -r requirements.txt
cp .env.example .env  # Add your tokens
python bot.py
```

## 🙏 Acknowledgments

- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [discord.py](https://discordpy.readthedocs.io/) for Discord API integration
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized transcription
- [OpenAI GPT](https://openai.com/gpt-4) for Q&A capabilities

