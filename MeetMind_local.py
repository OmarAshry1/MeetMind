import discord
from discord.ext import commands, voice_recv
import datetime
import os
import wave
import asyncio
import threading
import io
from faster_whisper import WhisperModel
from docx import Document
from collections import defaultdict
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import logging
import openai
from openai import AsyncOpenAI
import json
from flask import Flask, jsonify
import threading

# Initialize Flask app for healthcheck
app = Flask(__name__)

@app.route('/')
def healthcheck():
    """Healthcheck endpoint for Railway."""
    return jsonify({
        "status": "healthy",
        "bot": "MeetMind Discord Bot",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Detailed health endpoint."""
    return jsonify({
        "status": "healthy",
        "bot": "MeetMind Discord Bot",
        "features": [
            "Live transcription",
            "Multi-language support", 
            "Document generation",
            "AI Q&A about meetings"
        ],
        "timestamp": datetime.datetime.now().isoformat()
    })

def run_flask():
    """Run Flask server in a separate thread."""
    app.run(host='0.0.0.0', port=8000, debug=False)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


model = WhisperModel("small", device="cpu", compute_type="int8")


SUPPORTED_LANGUAGES = {
   
    "english": "en", "en": "en",
    "spanish": "es", "es": "es", "español": "es",
    "french": "fr", "fr": "fr", "français": "fr",
    "german": "de", "de": "de", "deutsch": "de",
    "italian": "it", "it": "it", "italiano": "it",
    "portuguese": "pt", "pt": "pt", "português": "pt",
    "russian": "ru", "ru": "ru", "русский": "ru",
    "chinese": "zh", "zh": "zh", "中文": "zh", "mandarin": "zh",
    "japanese": "ja", "ja": "ja", "日本語": "ja",
    "korean": "ko", "ko": "ko", "한국어": "ko",
    "hindi": "hi", "hi": "hi", "हिंदी": "hi",
    
    
    "arabic": "ar", "ar": "ar", "العربية": "ar",
    
    
    "dutch": "nl", "nl": "nl", "nederlands": "nl",
    "swedish": "sv", "sv": "sv", "svenska": "sv",
    "norwegian": "no", "no": "no", "norsk": "no",
    "danish": "da", "da": "da", "dansk": "da",
    "finnish": "fi", "fi": "fi", "suomi": "fi",
    "polish": "pl", "pl": "pl", "polski": "pl",
    "czech": "cs", "cs": "cs", "čeština": "cs",
    "hungarian": "hu", "hu": "hu", "magyar": "hu",
    "greek": "el", "el": "el", "ελληνικά": "el",
    "turkish": "tr", "tr": "tr", "türkçe": "tr",
    "hebrew": "he", "he": "he", "עברית": "he",
    
   
    "thai": "th", "th": "th", "ไทย": "th",
    "vietnamese": "vi", "vi": "vi", "tiếng việt": "vi",
    "indonesian": "id", "id": "id", "bahasa indonesia": "id",
    "malay": "ms", "ms": "ms", "bahasa melayu": "ms",
    "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
    "bengali": "bn", "bn": "bn", "বাংলা": "bn",
    "urdu": "ur", "ur": "ur", "اردو": "ur",
    "persian": "fa", "fa": "fa", "فارسی": "fa", "farsi": "fa",
    
   
    "auto": "auto", "automatic": "auto", "detect": "auto"
}

def get_language_code(language_input):
    """Convert language input to standardized language code."""
    if not language_input:
        return "en"
    
    lang_lower = language_input.lower().strip()
    return SUPPORTED_LANGUAGES.get(lang_lower, "en")

def get_language_display_name(language_code):
    """Get display name for language code."""
    lang_names = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "ar": "Arabic",
        "nl": "Dutch", "sv": "Swedish", "no": "Norwegian", "da": "Danish",
        "fi": "Finnish", "pl": "Polish", "cs": "Czech", "hu": "Hungarian",
        "el": "Greek", "tr": "Turkish", "he": "Hebrew", "th": "Thai",
        "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "ta": "Tamil",
        "bn": "Bengali", "ur": "Urdu", "fa": "Persian", "auto": "Auto-detect"
    }
    return lang_names.get(language_code, language_code.upper())


active_meetings = {}


class TranscriptionSink(voice_recv.AudioSink):
    def __init__(self, meeting, bot_instance, language="auto"):
        super().__init__()
        self.meeting = meeting
        self.bot = bot_instance
        self.language = language
        self.buffers = defaultdict(bytearray)
        self.last_time = defaultdict(lambda: datetime.datetime.now())
        self.processing = defaultdict(bool)
        
    def wants_opus(self) -> bool:
        return False
    
    def cleanup(self):
        self.buffers.clear()
        self.last_time.clear()
        self.processing.clear()
    
    def write(self, user, data: voice_recv.VoiceData):
        """Called by voice_recv when audio arrives (must be sync)."""
        if not data.pcm or not user:
            return
            
        uid = user.id
        
        
        self.buffers[uid].extend(data.pcm)
        
        now = datetime.datetime.now()
        elapsed = (now - self.last_time[uid]).total_seconds()
        
       
        if elapsed >= 7.0 and len(self.buffers[uid]) > 22400 and not self.processing[uid]:  # Increased minimum buffer size for better accuracy
            pcm_data = bytes(self.buffers[uid])
            self.buffers[uid] = bytearray()
            self.last_time[uid] = now
            self.processing[uid] = True
            
            
            def schedule_transcription():
                try:
                    # Get the bot's event loop
                    loop = self.bot.loop
                    if loop and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.process_audio_async(user, pcm_data), 
                            loop
                        )
                except Exception as e:
                    logger.error(f"Error scheduling transcription: {e}")
                    self.processing[uid] = False
            
            
            threading.Thread(target=schedule_transcription, daemon=True).start()
    
    async def process_audio_async(self, user, pcm_data):
        """Process audio data asynchronously."""
        uid = user.id
        try:
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(pcm_data)
            
            wav_buffer.seek(0)
            
          
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.transcribe_audio, wav_buffer)
            
            if text and text.strip():
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                speaker = user.display_name
                
                
                entry = {
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'text': text.strip(),
                    'user_id': uid
                }
                self.meeting["log"].append(entry)
                
                
                channel = self.meeting["channel"]
                if channel:
                    try:
                        await channel.send(f"[{timestamp}] **{speaker}**: {text.strip()}")
                    except discord.errors.NotFound:
                        logger.warning("Text channel not found, meeting may have ended")
                    except Exception as e:
                        logger.error(f"Error sending message to channel: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing audio for user {user.display_name}: {e}")
        finally:
            self.processing[uid] = False
    
    def transcribe_audio(self, wav_buffer):
        """Transcribe audio using Whisper model with language support."""
        try:
            
            language_code = None if self.language == "auto" else self.language
            
            segments, info = model.transcribe(
                wav_buffer, 
                language=language_code,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                temperature=0.0,
                beam_size=5
            )
            
            text_segments = []
            for segment in segments:
                
                if segment.text.strip() and len(segment.text.strip()) > 1:
                    text_segments.append(segment.text.strip())
            
            return " ".join(text_segments)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


async def get_meeting_context_from_channel(channel):
    """Get formatted meeting context from channel message history."""
    try:
        # Check if this is a meeting transcription channel
        if not (channel.name.startswith('meeting-transcription-') or 
                ('transcription' in channel.topic.lower() if channel.topic else False)):
            return None
        
        messages = []
        
        async for message in channel.history(limit=500, oldest_first=True):
            # Skip bot system messages and commands
            if message.author.bot and not message.content.startswith('['):
                continue
            if message.content.startswith('!'):
                continue
                
            
            if message.content.startswith('[') and '**' in message.content and '**:' in message.content:
                messages.append(message.content)
        
        if not messages:
            return None
        
        
        context = "Meeting Transcript:\n"
        context += f"Channel: {channel.name}\n"
        context += f"Total Messages: {len(messages)}\n"
        context += "="*50 + "\n\n"
        
        
        for msg in messages:
            context += f"{msg}\n"
        
        return context
        
    except Exception as e:
        logger.error(f"Error getting meeting context from channel: {e}")
        return None

async def get_meeting_context(guild_id):
    """Get formatted meeting context for OpenAI (for active meetings)."""
    if guild_id not in active_meetings:
        return None
    
    meeting = active_meetings[guild_id]
    if not meeting["log"]:
        return None
    
   
    context = "Meeting Transcript:\n"
    context += f"Started: {meeting['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
    context += f"Language: {meeting.get('language_display', 'Auto-detect')}\n"
    context += "="*50 + "\n\n"
    
    for entry in meeting["log"]:
        context += f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}\n"
    
    return context

async def ask_openai_about_meeting(transcript_context, question):
    """Ask OpenAI about the meeting content."""
    try:
        system_prompt = """You are a helpful assistant that answers questions about meeting transcripts. 
        You should only answer questions based on the provided transcript content. 
        If the information is not available in the transcript, politely say so.
        Be concise and accurate in your responses.
        When referencing what someone said, include their name and approximate time if available."""
        
        user_prompt = f"""Here is the meeting transcript:

{transcript_context}

Question: {question}

Please answer based only on the information provided in the transcript above."""

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"


async def create_transcript_document(meeting_log, format_type="docx"):
    """Create a transcript document in the specified format."""
    if format_type == "docx":
        return await create_word_document(meeting_log)
    elif format_type == "pdf":
        return await create_pdf_document(meeting_log)
    else:
        return await create_text_document(meeting_log)

async def create_word_document(meeting_log):
    """Create a Word document transcript."""
    doc = Document()
    doc.add_heading("Meeting Transcript", 0)
    doc.add_paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("=" * 50)
    
    for entry in meeting_log:
        doc.add_paragraph(f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}")
    
    filename = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(filename)
    return filename

async def create_pdf_document(meeting_log):
    """Create a PDF document transcript."""
    filename = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    
    story.append(Paragraph("Meeting Transcript", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    
    for entry in meeting_log:
        line = f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}"
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 6))
    
    doc.build(story)
    return filename

async def create_text_document(meeting_log):
    """Create a text file transcript."""
    filename = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Meeting Transcript\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for entry in meeting_log:
            f.write(f"[{entry['timestamp']}] {entry['speaker']}: {entry['text']}\n")
    
    return filename


@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_voice_state_update(member, before, after):
    """Handle voice state changes to clean up meetings when everyone leaves."""
    
    for guild_id, meeting in list(active_meetings.items()):
        if meeting["vc"] and meeting["vc"].channel:
           
            members_in_channel = [m for m in meeting["vc"].channel.members if not m.bot]
            
            
            if len(members_in_channel) == 0:
                logger.info(f"Auto-ending meeting in guild {guild_id} - no members left")
                await auto_end_meeting(guild_id)

async def auto_end_meeting(guild_id):
    """Automatically end a meeting when all participants leave."""
    if guild_id not in active_meetings:
        return
    
    meeting = active_meetings.pop(guild_id)
    
    try:
        # Clean up the sink
        if hasattr(meeting.get("sink"), 'cleanup'):
            meeting["sink"].cleanup()
        
        # Stop voice connection
        if meeting["vc"]:
            await meeting["vc"].disconnect()
        
        # Generate and send transcript if there are entries
        if meeting["log"]:
            filename = await create_transcript_document(meeting["log"], "pdf")
            channel = meeting["channel"]
            
            if channel:
                try:
                    with open(filename, 'rb') as f:
                        file = discord.File(f, filename)
                        await channel.send("📄 Meeting ended automatically. Here's the transcript:", file=file)
                    
                    # Clean up file
                    os.remove(filename)
                except Exception as e:
                    logger.error(f"Error sending auto-generated transcript: {e}")
        
    except Exception as e:
        logger.error(f"Error in auto_end_meeting: {e}")

# ==== Commands ====
@bot.command(name='start_meeting', aliases=['start', 'begin_meeting'])
async def start_meeting(ctx, *, language:str|None=None):
    """Start a new meeting with transcription in specified language.
    
    Usage: !start_meeting [language]
    Examples: 
        !start_meeting
        !start_meeting english
        !start_meeting arabic
        !start_meeting auto
    """
    # Check if user is in a voice channel
    if ctx.author.voice is None:
        await ctx.send("❌ You must be in a voice channel to start a meeting.")
        return
    
    guild_id = ctx.guild.id
    
    # Check if a meeting is already active
    if guild_id in active_meetings:
        await ctx.send("❌ A meeting is already active in this server. Use `!end_meeting` to stop the current meeting.")
        return
    
    # Process language input
    language_code = get_language_code(language)
    language_display = get_language_display_name(language_code)
    
    guild = ctx.guild
    
    try:
        # Create a dedicated text channel for transcription
        channel_name = f"meeting-transcription-{datetime.datetime.now().strftime('%m%d-%H%M')}"
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=True, send_messages=False),
            guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True)
        }
        
        text_channel = await guild.create_text_channel(
            channel_name, 
            overwrites=overwrites,
            topic=f"Live transcription for meeting started by {ctx.author.display_name} | Language: {language_display}"
        )
        
        await text_channel.send(
            f"🎙️ **Meeting Transcription Started**\n"
            f"Started by: {ctx.author.mention}\n"
            f"Voice Channel: {ctx.author.voice.channel.name}\n"
            f"Language: {language_display}\n"
            f"💡 Use `!ask <question>` to ask about the meeting content!\n"
            + "="*50
        )
        
    except discord.Forbidden:
        # Fallback if bot cannot create channels
        text_channel = ctx.channel
        await ctx.send(f"⚠️ Missing permissions to create channels. Using this channel for transcription.\n🌐 Language: {language_display}")
    
    # Connect to voice channel
    vc_channel = ctx.author.voice.channel
    try:
        voice_client = await vc_channel.connect(cls=voice_recv.VoiceRecvClient)
    except Exception as e:
        await ctx.send(f"❌ Failed to connect to voice channel: {e}")
        return
    
    # Create meeting record
    meeting = {
        "channel": text_channel,
        "log": [],
        "vc": voice_client,
        "start_time": datetime.datetime.now(),
        "started_by": ctx.author.id,
        "language": language_code,
        "language_display": language_display
    }
    
    active_meetings[guild_id] = meeting
    
    # Set up audio transcription with language support
    sink = TranscriptionSink(meeting, bot, language_code)
    meeting["sink"] = sink  # Store reference for cleanup
    voice_client.listen(sink)
    
    await ctx.send(f"✅ Meeting started! Live transcriptions in **{language_display}** will appear in {text_channel.mention}")

@bot.command(name='end_meeting', aliases=['end', 'stop_meeting'])
async def end_meeting(ctx, format_type="pdf"):
    """End the current meeting and generate transcript."""
    guild_id = ctx.guild.id
    
    if guild_id not in active_meetings:
        await ctx.send("❌ No active meeting found in this server.")
        return
    
    meeting = active_meetings.pop(guild_id)
    
    try:
        # Clean up the sink
        if hasattr(meeting.get("sink"), 'cleanup'):
            meeting["sink"].cleanup()
        
        # Disconnect from voice
        if meeting["vc"]:
            await meeting["vc"].disconnect()
        
        # Generate transcript document
        if meeting["log"]:
            # Validate format type
            if format_type.lower() not in ["pdf", "docx", "txt"]:
                format_type = "pdf"
            
            filename = await create_transcript_document(meeting["log"], format_type.lower())
            
            # Send transcript to the channel
            channel = meeting["channel"]
            duration = datetime.datetime.now() - meeting["start_time"]
            
            embed = discord.Embed(
                title="📄 Meeting Ended",
                description=f"Meeting duration: {str(duration).split('.')[0]}",
                color=0x00ff00
            )
            embed.add_field(name="Total Messages", value=len(meeting["log"]), inline=True)
            embed.add_field(name="Format", value=format_type.upper(), inline=True)
            
            try:
                with open(filename, 'rb') as f:
                    file = discord.File(f, filename)
                    await channel.send(embed=embed, file=file)
                
                # Clean up file
                os.remove(filename)
                await ctx.send("✅ Meeting ended successfully! Transcript has been generated.")
                
            except Exception as e:
                logger.error(f"Error sending transcript file: {e}")
                await ctx.send("✅ Meeting ended, but there was an error sending the transcript file.")
        else:
            await ctx.send("✅ Meeting ended. No transcription data was recorded.")
            
    except Exception as e:
        logger.error(f"Error ending meeting: {e}")
        await ctx.send("❌ Error ending meeting. Please try again.")

@bot.command(name='ask')
async def ask_meeting_question(ctx, *, question):
    """Ask a question about the meeting transcript in the current channel.
    
    Usage: !ask <your question>
    Example: !ask What did John say about the budget?
    
    This command works in:
    - Active meeting transcription channels
    - Completed meeting transcription channels (using message history)
    """
    if not question or len(question.strip()) == 0:
        await ctx.send("❌ Please provide a question to ask about the meeting.")
        return
    
    # First try to get context from active meeting
    guild_id = ctx.guild.id
    transcript_context = None
    
    if guild_id in active_meetings:
        active_meeting = active_meetings[guild_id]
        if active_meeting["channel"].id == ctx.channel.id:
            # User is asking in the active meeting channel
            transcript_context = await get_meeting_context(guild_id)
    
    # If no active meeting context, try to get from channel history
    if not transcript_context:
        transcript_context = await get_meeting_context_from_channel(ctx.channel)
    
    if not transcript_context:
        await ctx.send(
            "❌ No meeting transcript found in this channel.\n"
            "💡 Use this command in:\n"
            "• An active meeting transcription channel\n" 
            "• A completed meeting transcription channel\n"
            "• Start a meeting with `!start_meeting` first"
        )
        return
    
    # Check if transcript is too long and truncate if needed (to avoid token limits)
    max_length = 12000  # Approximately 3000 tokens
    if len(transcript_context) > max_length:
        lines = transcript_context.split('\n')
        header_lines = lines[:5]  # Keep header info
        transcript_lines = [line for line in lines[5:] if line.startswith('[')]
        
        # Take recent messages if truncation is needed
        if len('\n'.join(transcript_lines)) > max_length - 500:
            transcript_lines = transcript_lines[-100:]  # Last 100 messages
            header_lines.append("⚠️ [Transcript truncated to recent messages due to length]")
        
        transcript_context = '\n'.join(header_lines + transcript_lines)
    
    # Show typing indicator
    async with ctx.typing():
        try:
            # Get answer from OpenAI
            answer = await ask_openai_about_meeting(transcript_context, question)
            
            # Create embed for the response
            embed = discord.Embed(
                title="🤖 Meeting Q&A",
                color=0x00ff99
            )
            embed.add_field(name="❓ Question", value=question, inline=False)
            embed.add_field(name="💬 Answer", value=answer, inline=False)
            
            # Add context info
            if guild_id in active_meetings and active_meetings[guild_id]["channel"].id == ctx.channel.id:
                embed.set_footer(text="Based on active meeting transcript")
            else:
                embed.set_footer(text="Based on channel message history")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error in ask command: {e}")
            await ctx.send("❌ Sorry, I encountered an error while processing your question. Please try again.")

@bot.command(name='meeting_status', aliases=['status'])
async def meeting_status(ctx):
    """Check the status of the current meeting."""
    guild_id = ctx.guild.id
    
    if guild_id not in active_meetings:
        await ctx.send("❌ No active meeting in this server.")
        return
    
    meeting = active_meetings[guild_id]
    duration = datetime.datetime.now() - meeting["start_time"]
    
    embed = discord.Embed(
        title="🎙️ Meeting Status",
        color=0x00ff00
    )
    embed.add_field(name="Duration", value=str(duration).split('.')[0], inline=True)
    embed.add_field(name="Messages Transcribed", value=len(meeting["log"]), inline=True)
    embed.add_field(name="Language", value=meeting.get("language_display", "Auto-detect"), inline=True)
    embed.add_field(name="Channel", value=meeting["channel"].mention, inline=True)
    
    if meeting["vc"] and meeting["vc"].channel:
        members = [m.display_name for m in meeting["vc"].channel.members if not m.bot]
        embed.add_field(name="Participants", value=", ".join(members) or "None", inline=False)
    
    await ctx.send(embed=embed)

# ==== Error Handling ====
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        if ctx.command.name == "ask":
            await ctx.send("❌ Please provide a question. Usage: `!ask <your question>`")
        else:
            await ctx.send(f"❌ Missing required argument for command `{ctx.command.name}`")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"❌ An error occurred: {str(error)}")

# ==== Help Command ====
@bot.command(name='meeting_help', aliases=['help_meeting'])
async def meeting_help(ctx):
    """Show help for meeting commands."""
    embed = discord.Embed(
        title="🎙️ Meeting Transcription Bot Help",
        description="Commands for managing meeting transcriptions and Q&A",
        color=0x0099ff
    )
    
    embed.add_field(
        name="!start_meeting [language]", 
        value="Start meeting with live transcription in specified language\nExamples: `!start_meeting arabic`, `!start_meeting english`", 
        inline=False
    )
    embed.add_field(
        name="!end_meeting [format]", 
        value="End meeting and generate transcript (format: pdf, docx, txt)", 
        inline=False
    )
    embed.add_field(
        name="!ask <question>", 
        value="Ask a question about meeting transcript in current channel\nWorks during and after meetings in transcription channels\nExample: `!ask What did Sarah say about the deadline?`", 
        inline=False
    )
    embed.add_field(
        name="!meeting_status", 
        value="Check current meeting status", 
        inline=False
    )
    embed.add_field(
        name="!languages", 
        value="Show supported languages for transcription", 
        inline=False
    )
    embed.add_field(
        name="!meeting_help", 
        value="Show this help message", 
        inline=False
    )
    
    embed.set_footer(text="Bot requires permissions: Connect, Speak, Use Voice Activity")
    
    await ctx.send(embed=embed)

@bot.command(name='languages', aliases=['supported_languages', 'lang'])
async def show_languages(ctx):
    """Show supported languages for transcription."""
    embed = discord.Embed(
        title="🌐 Supported Languages",
        description="Languages available for meeting transcription",
        color=0x00ff99
    )
    
    # Group languages by region for better organization
    languages_by_region = {
        "🌍 European": ["english", "spanish", "french", "german", "italian", "portuguese", "russian", "dutch", "swedish", "norwegian", "danish", "finnish", "polish", "czech", "hungarian", "greek"],
        "🏛️ Middle Eastern": ["arabic", "hebrew", "persian", "turkish", "urdu"],
        "🏮 Asian": ["chinese", "japanese", "korean", "hindi", "thai", "vietnamese", "indonesian", "malay", "tamil", "bengali"],
        "🤖 Special": ["auto (automatic detection)"]
    }
    
    for region, langs in languages_by_region.items():
        lang_list = ", ".join(f"`{lang}`" for lang in langs)
        embed.add_field(name=region, value=lang_list, inline=False)
    
    embed.add_field(
        name="📝 Usage Examples",
        value="`!start_meeting arabic`\n`!start_meeting english`\n`!start_meeting auto`",
        inline=False
    )
    
    embed.set_footer(text="Language detection is automatic if not specified")
    
    await ctx.send(embed=embed)

# ==== Run Bot ====
if __name__ == "__main__":
    # Check for required environment variables
    discord_token = os.getenv('DISCORD_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not discord_token:
        print("ERROR: DISCORD_TOKEN not found in environment variables!")
        print("Please create a .env file with your bot token:")
        print("DISCORD_TOKEN=your_token_here")
        exit(1)
    
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("The !ask command will not work without OpenAI API key.")
        print("Add to your .env file: OPENAI_API_KEY=your_api_key_here")
    
    print("🤖 Starting Discord Meeting Bot...")
    print("📋 Features enabled:")
    print("   ✅ Live transcription")
    print("   ✅ Multi-language support") 
    print("   ✅ Document generation (PDF, DOCX, TXT)")
    if openai_api_key:
        print("   ✅ AI Q&A about meetings")
    else:
        print("   ❌ AI Q&A (missing OpenAI API key)")
    
    print("🌐 Starting Flask healthcheck server...")
    # Start Flask server in a separate thread (non-daemon so it stays alive)
    flask_thread = threading.Thread(target=run_flask, daemon=False)
    flask_thread.start()
    
    # Give Flask server a moment to start up
    import time
    time.sleep(2)
    print("✅ Flask server started on port 8000")
    
    print("🚀 Bot starting...")
    try:
        bot.run(discord_token)
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        exit(1)