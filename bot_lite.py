"""
╔═══════════════════════════════════════════════════════════╗
║  JARVIS - Discord Voice AI Assistant (LITE VERSION)      ║
║  Created by: Cob / OneRaap Hosting                       ║
║  GitHub: https://github.com/CobCob047/jarvis-lite  ║
╚═══════════════════════════════════════════════════════════╝

🎙️ LITE VERSION FEATURES:
✅ Real-time voice transcription (Whisper large-v3)
✅ Emotion detection from voice tone
✅ Stats & leaderboards (who talks most, top words, etc.)
✅ Music playback (Spotify + YouTube integration)
✅ AI responses powered by Ollama
✅ Full conversation logging & search
✅ 200+ voice commands

💎 PREMIUM VERSION AVAILABLE:
🔥 17 Psychological Warfare Features
🔥 Advanced Behavioral Analysis
🔥 Loyalty Tracking & Social Dynamics
🔥 Relationship Destroyer Suite
🔥 Custom Enterprise Features
🔥 Priority Support

Contact: [your-email] for premium access
"""

import discord

# Suppress noisy voice recv logs
import logging
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.WARNING)
from datetime import datetime, timedelta
from discord.ext import commands, voice_recv
from aiohttp import web
import asyncio
import sqlite3
from pysqlcipher3 import dbapi2 as sqlcipher
import numpy as np
from faster_whisper import WhisperModel
import librosa
import pvporcupine
import torch
import os
with open(".db_password", "r") as f:
    DB_PASSWORD = f.read().strip()
from gtts import gTTS
import tempfile
import hashlib

# TTS Cache
TTS_CACHE = {}
TTS_CACHE_DIR = "/tmp/tts_cache"
ATTACHMENTS_DIR = "/root/discord_stenographer/attachments"
import os
os.makedirs(TTS_CACHE_DIR, exist_ok=True)
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
import json
from pathlib import Path
import parselmouth
from parselmouth.praat import call
import yt_dlp

# Logging setup
import logging
from logging.handlers import RotatingFileHandler

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
logger = logging.getLogger('jarvis')
logger.setLevel(logging.INFO)

# File handler with rotation (10MB per file, keep 5 files)
file_handler = RotatingFileHandler(
    'logs/jarvis.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
logger.info("Loading Whisper model...")
whisper_model = WhisperModel("distil-large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
logger.info("Whisper model loaded!")

# Initialize Porcupine for instant wake word detection
PORCUPINE_ACCESS_KEY = "7RAlvnh8A2Yd/Jl2lZu1DCVLGas9rmX8GYa0XMsYGWE+CcbCqTZ1YA=="
porcupine = None
try:
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keywords=['jarvis']
    )
    logger.info(f"Porcupine wake word detector loaded! Frame length: {porcupine.frame_length}")
except Exception as e:
    logger.warning(f"Porcupine failed to load: {e}")
    porcupine = None

OLLAMA_URL = "http://localhost:11434"

# Spotify setup
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
    SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri='http://localhost:9000/callback',
            scope='user-library-read playlist-read-private'
        ))
        print("✅ Spotify connected!")
    else:
        sp = None
except:
    sp = None

# Retry wrapper for network calls

def number_to_words(n):
    """Convert numbers to words for TTS"""
    if n < 20:
        return ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen'][n]
    elif n < 100:
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        return tens[n // 10] + ('' if n % 10 == 0 else ' ' + number_to_words(n % 10))
    elif n < 1000:
        return number_to_words(n // 100) + ' hundred' + ('' if n % 100 == 0 else ' ' + number_to_words(n % 100))
    elif n < 1000000:
        return number_to_words(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + number_to_words(n % 1000))
    else:
        return str(n)

async def retry_async(func, max_retries=3, delay=1):
    """Retry async function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            await asyncio.sleep(delay * (2 ** attempt))
    return None

# Bot metadata
BOT_VERSION = "2.0"
bot_start_time = None


# Load database password
#     DB_PASSWORD = f.read().strip()

def get_db_connection():
    """Get encrypted database connection"""
    conn = sqlcipher.connect('transcriptions_encrypted.db')
    conn.execute(f"PRAGMA key='{DB_PASSWORD}'")
    conn.execute(f"PRAGMA key='{DB_PASSWORD}'")
    return conn

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

voice_clients = {}

last_packet_time = {}  # Track last packet received per guild
opus_monitoring_active = True

music_queues = {}
currently_playing = {}  # Track what song is currently playing per guild
muted_users = {}
conversation_history = {}
voice_settings = {}
personality_mode = {}  # Store personality per guild  # Store voice/personality settings per guild
volume_settings = {}

loop_settings = {}  # guild_id -> repeat settings
last_actions = {}  # guild_id -> last action
active_reminders = []  # reminder tasks
transcription_language = {}  # guild_id -> language code
wake_word_strict = True  # strict wake word mode

# User blocklist - usernames blocked from music commands
MUSIC_BLOCKLIST = ['alex', 'alikalee']  # alikalee0955 is allowed

# Debug mode toggle
DEBUG_MODE = True  # Start enabled

# Multi-server sync (owner only)
SYNC_ENABLED = False
SYNC_GUILDS = []  # List of guild IDs to sync  # guild_id -> default_volume

class AudioSink(voice_recv.AudioSink):
    def __init__(self, guild_id, channel_id):
        super().__init__()
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.user_buffers = {}
    
    def write(self, user, data):
        if user is None:
            return
        if user.id not in self.user_buffers:
            self.user_buffers[user.id] = {'user': user, 'buffer': []}
        self.user_buffers[user.id]['buffer'].append(data.pcm)
    
    def wants_opus(self):
        return False
    
    def cleanup(self):
        pass

# Voice Biometrics System
VOICE_PROFILES_FILE = Path('voice_profiles.json')

def load_voice_profiles():
    """Load existing voice profiles"""
    if VOICE_PROFILES_FILE.exists():
        with open(VOICE_PROFILES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_voice_profiles(profiles):
    """Save voice profiles to disk"""
    with open(VOICE_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)

def analyze_voice_features(audio_data, sample_rate=48000):
    """Extract voice biometric features from audio"""
    try:
        # Convert audio to proper format for analysis
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Resample to 16kHz for Praat analysis
        audio_16k = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
        
        # Save temporarily for Praat
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_16k, 16000)
            
            # Analyze with Praat
            sound = parselmouth.Sound(tmp.name)
            
            # Extract features
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
            
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")
            mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
            
            # Get formants (vocal tract characteristics - unique per person)
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
            
            # Clean up
            import os
            os.unlink(tmp.name)
            
            return {
                'pitch_mean': float(mean_pitch) if mean_pitch else 0,
                'intensity_mean': float(mean_intensity) if mean_intensity else 0,
                'formant_f1': float(f1) if f1 else 0,
                'formant_f2': float(f2) if f2 else 0
            }
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return None

def detect_emotion_from_voice(features):
    """Detect emotional state from voice features"""
    if not features:
        return "neutral"
    
    pitch = features.get('pitch_mean', 0)
    intensity = features.get('intensity_mean', 0)
    
    # Simple heuristics (can be improved with ML)
    if pitch > 200 and intensity > 70:
        return "excited/angry"
    elif pitch < 120 and intensity < 50:
        return "sad/tired"
    elif intensity > 75:
        return "energetic"
    elif pitch > 180:
        return "happy/stressed"
    else:
        return "calm"

def detect_intoxication(features, baseline):
    """Detect potential intoxication from speech patterns"""
    if not features or not baseline:
        return False, 0
    
    # Compare to baseline
    pitch_variance = abs(features['pitch_mean'] - baseline['pitch_mean'])
    intensity_variance = abs(features['intensity_mean'] - baseline['intensity_mean'])
    
    # Intoxication typically shows: lower pitch, more variance, lower intensity
    intox_score = 0
    if features['pitch_mean'] < baseline['pitch_mean'] * 0.9:
        intox_score += 30
    if intensity_variance > 20:
        intox_score += 25
    if pitch_variance > 40:
        intox_score += 25
    
    return intox_score > 50, intox_score

# Auto DJ System
AUTO_DJ_ENABLED = {}
CONVERSATION_ENERGY_HISTORY = {}

async def analyze_conversation_energy(guild_id):
    """Analyze recent conversation energy level"""
    try:
        from datetime import datetime, timedelta
        
        # Get last 5 minutes of transcriptions
        cutoff = (datetime.now() - timedelta(minutes=5)).isoformat()
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""SELECT transcription, timestamp FROM transcriptions 
                    WHERE guild_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 50""",
                 (str(guild_id), cutoff))
        results = c.fetchall()
        conn.close()
        
        if not results:
            return 0
        
        # Calculate energy metrics
        total_words = sum(len(r[0].split()) for r in results)
        messages_per_minute = len(results) / 5
        
        # Check for high-energy indicators
        exclamations = sum(r[0].count('!') for r in results)
        all_caps_words = sum(1 for r in results for word in r[0].split() if word.isupper() and len(word) > 2)
        
        # Energy score (0-100)
        energy = min(100, (messages_per_minute * 10) + (exclamations * 5) + (all_caps_words * 3))
        
        return int(energy)
        
    except Exception as e:
        print(f"Energy analysis error: {e}")
        return 0

async def auto_dj_monitor(guild_id):
    """Background task to monitor conversation and auto-play music"""
    await asyncio.sleep(60)  # Wait for bot to stabilize
    
    while AUTO_DJ_ENABLED.get(guild_id, False):
        try:
            energy = await analyze_conversation_energy(guild_id)
            
            # Store energy history
            if guild_id not in CONVERSATION_ENERGY_HISTORY:
                CONVERSATION_ENERGY_HISTORY[guild_id] = []
            CONVERSATION_ENERGY_HISTORY[guild_id].append(energy)
            if len(CONVERSATION_ENERGY_HISTORY[guild_id]) > 10:
                CONVERSATION_ENERGY_HISTORY[guild_id].pop(0)
            
            # Check if music should auto-play
            if guild_id in voice_clients and 'vc' in voice_clients[guild_id]:
                vc = voice_clients[guild_id]['vc']
                
                # If not playing and energy is high, play upbeat music
                if not vc.is_playing() and energy > 60:
                    print(f"🎵 Auto DJ: High energy ({energy}), playing upbeat music")
                    
                    # Search for upbeat songs
                    upbeat_songs = [
                        "upbeat party music",
                        "energetic electronic music", 
                        "hype rap music",
                        "pump up music"
                    ]
                    import random
                    query = random.choice(upbeat_songs)
                    
                    song = await search_youtube(query, guild_id)
                    if song and guild_id in music_queues:
                        music_queues[guild_id].append(song)
                        if not vc.is_playing():
                            await play_next(guild_id)
                
                # If not playing and energy is low, play chill music
                elif not vc.is_playing() and energy < 20:
                    print(f"🎵 Auto DJ: Low energy ({energy}), playing chill music")
                    
                    chill_songs = [
                        "lofi hip hop chill",
                        "ambient relaxing music",
                        "chill instrumental",
                        "study music calm"
                    ]
                    import random
                    query = random.choice(chill_songs)
                    
                    song = await search_youtube(query, guild_id)
                    if song and guild_id in music_queues:
                        music_queues[guild_id].append(song)
                        if not vc.is_playing():
                            await play_next(guild_id)
            
            await asyncio.sleep(120)  # Check every 2 minutes
            
        except Exception as e:
            print(f"Auto DJ error: {e}")
            await asyncio.sleep(60)

# Real-Time Fact Checker System
FACT_CHECKER_ENABLED = {}
FACT_CHECK_PATTERNS = [
    # Factual claim indicators
    r'the .* is',
    r'actually',
    r'in fact',
    r'\d{4}',  # Years
    r'tallest',
    r'biggest',
    r'fastest',
    r'first',
    r'invented',
    r'discovered',
    r'percent',
    r'million',
    r'billion',
]

async def detect_factual_claim(text):
    """Detect if transcription contains a factual claim worth checking"""
    import re
    text_lower = text.lower()
    
    # Check for factual claim patterns
    for pattern in FACT_CHECK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    # Check for definitive statements
    definitive_words = ['is the', 'was the', 'are the', 'were the', 'always', 'never', 'all', 'none']
    if any(word in text_lower for word in definitive_words):
        return True
    
    return False

async def fact_check_claim(claim, guild_id):
    """Fact check a claim using web search"""
    try:
        import httpx
        
        # Use web search to verify
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Simple DuckDuckGo instant answer API
            response = await client.get(
                f"https://api.duckduckgo.com/?q={claim}&format=json&no_html=1"
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we got a definitive answer
                if data.get('AbstractText'):
                    answer = data['AbstractText']
                    
                    # Simple contradiction detection
                    claim_lower = claim.lower()
                    answer_lower = answer.lower()
                    
                    # Look for contradictions
                    if 'not' in answer_lower or 'false' in answer_lower or 'incorrect' in answer_lower:
                        return True, answer[:200]  # Claim is FALSE
                    
                return False, None  # Can't verify or claim seems OK
                
        return False, None
        
    except Exception as e:
        print(f"Fact check error: {e}")
        return False, None

async def fact_checker_monitor(guild_id):
    """Background task to monitor and fact-check statements"""
    await asyncio.sleep(30)
    
    last_checked_id = 0
    
    while FACT_CHECKER_ENABLED.get(guild_id, False):
        try:
            # Get recent transcriptions
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT rowid, username, transcription FROM transcriptions 
                        WHERE guild_id = ? AND rowid > ?
                        ORDER BY rowid DESC LIMIT 5""",
                     (str(guild_id), last_checked_id))
            results = c.fetchall()
            conn.close()
            
            for row in results:
                rowid, username, transcription = row
                last_checked_id = max(last_checked_id, rowid)
                
                # Check if this contains a factual claim
                if await detect_factual_claim(transcription):
                    print(f"🔍 Checking claim from {username}: {transcription}")
                    
                    # Fact check it
                    is_false, correction = await fact_check_claim(transcription, guild_id)
                    
                    if is_false and correction:
                        # INTERRUPT with correction!
                        if guild_id in voice_clients and 'vc' in voice_clients[guild_id]:
                            vc = voice_clients[guild_id]['vc']
                            
                            # Shorten correction to first 100 chars max
                            short_correction = correction[:100] if len(correction) > 100 else correction
                            interrupt_text = f"Actually {username}, {short_correction}"
                            print(f"🚨 FACT CHECK INTERRUPT: {interrupt_text}")
                            
                            # Speak the correction
                            await asyncio.to_thread(speak_text, interrupt_text, vc, guild_id)
            
            await asyncio.sleep(3)  # Check every 3 seconds
            
        except Exception as e:
            print(f"Fact checker monitor error: {e}")
            await asyncio.sleep(5)

# PSYCHOLOGICAL WARFARE SYSTEMS
GASLIGHTING_DETECTOR_ENABLED = {}
MANIPULATION_TRACKER_ENABLED = {}
STATEMENT_HISTORY = {}  # Track what people have said

# Gaslighting patterns


# Manipulation tactics


async def detect_manipulation(statement, username):
    """Detect manipulation tactics in statement"""
    statement_lower = statement.lower()
    detected_tactics = []
    
    for tactic, patterns in MANIPULATION_PATTERNS.items():
        for pattern in patterns:
            if pattern in statement_lower:
                detected_tactics.append(tactic.replace('_', ' '))
                break
    
    if detected_tactics:
        tactics_str = ", ".join(detected_tactics)
        return True, f"{username} is using {tactics_str}"
    
    return False, None

async def manipulation_monitor(guild_id):
    """Monitor for manipulation tactics"""
    await asyncio.sleep(30)
    last_checked_id = 0
    
    while MANIPULATION_TRACKER_ENABLED.get(guild_id, False):
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT rowid, username, transcription FROM transcriptions 
                        WHERE guild_id = ? AND rowid > ?
                        ORDER BY rowid DESC LIMIT 5""",
                     (str(guild_id), last_checked_id))
            results = c.fetchall()
            conn.close()
            
            for row in results:
                rowid, username, transcription = row
                last_checked_id = max(last_checked_id, rowid)
                
                # Check for manipulation
                is_manipulative, tactic = await detect_manipulation(transcription, username)
                
                if is_manipulative and tactic:
                    print(f"⚠️  MANIPULATION DETECTED: {tactic}")
                    
                    # CALL IT OUT
                    if guild_id in voice_clients and 'vc' in voice_clients[guild_id]:
                        vc = voice_clients[guild_id]['vc']
                        callout = f"Warning: {tactic}"
                        await asyncio.to_thread(speak_text, callout, vc, guild_id)
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Manipulation monitor error: {e}")
            await asyncio.sleep(5)

# RELATIONSHIP DESTROYER FEATURES
SHIT_TALK_TRACKER_ENABLED = {}


async def detect_voice_stress_lying(features, baseline):
    """Enhanced lie detection using voice stress"""
    if not features or not baseline:
        return False, 0
    
    # Voice stress indicators
    stress_score = 0
    
    # Higher pitch = stress/lying
    if features['pitch_mean'] > baseline['pitch_mean'] * 1.15:
        stress_score += 30
    
    # Intensity variation = nervousness
    if abs(features['intensity_mean'] - baseline['intensity_mean']) > 15:
        stress_score += 25
    
    # Formant changes = vocal tension
    if features.get('formant_f1', 0) != 0 and baseline.get('formant_f1', 0) != 0:
        f1_diff = abs(features['formant_f1'] - baseline['formant_f1'])
        if f1_diff > 100:
            stress_score += 20
    
    is_lying = stress_score > 50
    confidence = min(stress_score, 100)
    
    return is_lying, confidence


async def search_youtube(query, guild_id=None):
    """Search YouTube with retry"""
    async def _search():
        with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
            info = ydl.extract_info(f"ytsearch:{query}", download=False)
            if 'entries' in info and len(info['entries']) > 0:
                video = info['entries'][0]
                return {'url': video['url'], 'title': video['title'], 'webpage_url': video['webpage_url']}
    
    try:
        return await retry_async(_search, max_retries=2)
    except Exception as e:
        print(f"YouTube error after retries: {e}")
        return None

async def play_next(guild_id):
    """Play next song"""
    if guild_id not in music_queues or not music_queues[guild_id]:
        return
    
    song = music_queues[guild_id].pop(0)
    currently_playing[guild_id] = song  # Track what's playing
    vc = voice_clients[guild_id]['vc']
    
    try:
        default_vol = volume_settings.get(guild_id, 0.06)
        source = discord.PCMVolumeTransformer(
            discord.FFmpegPCMAudio(song['url'], **FFMPEG_OPTIONS),
            volume=default_vol
        )
        
        def after_playing(error):
            if error:
                print(f"Error: {error}")
            if guild_id in music_queues and music_queues[guild_id]:
                asyncio.run_coroutine_threadsafe(play_next(guild_id), bot.loop)
        
        vc.play(source, after=after_playing)
        print(f"🎵 Now playing: {song['title']}")
    except Exception as e:
        print(f"Play error: {e}")

async def handle_voice_command(text, guild, requestor):
    """Handle voice commands"""
    global SYNC_ENABLED, SYNC_GUILDS
    import re as regex_module
    text_lower = text.lower()
    guild_id = guild.id
    
    # Music blocklist check function
    def is_music_blocked(username):
        """Check if user is blocked from music commands"""
        username_lower = username.lower()
        for blocked in MUSIC_BLOCKLIST:
            if username_lower == blocked:
                return True
        return False
    
    # Check Discord soundboard FIRST - accept "soundname" or "play soundname"
    try:
        soundboard_sounds = await guild.fetch_soundboard_sounds()
        
        # Strip "play" if present
        search_text = text_lower.replace('play ', '').strip()
        
        # Direct match or partial match
        for sound in soundboard_sounds:
            sound_name_lower = sound.name.lower()
            if search_text == sound_name_lower or search_text in sound_name_lower or sound_name_lower in search_text:
                # Found a matching soundboard sound
                vc = voice_clients.get(guild_id, {}).get('vc')
                if vc and vc.channel:
                    try:
                        # Play soundboard sound using channel.send_sound()
                        await vc.channel.send_sound(sound)
                        return f"Playing soundboard: {sound.name}"
                    except Exception as e:
                        print(f"Soundboard play error: {e}")
                        return f"Couldn't play {sound.name}"
                return "Not in voice channel"
    except Exception as e:
        print(f"Soundboard check error: {e}")
        pass
    

    # Conversation summary

{conversation[:3000]}

Summary:"""
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": summary_prompt, "stream": False}
                )
            
            if response.status_code == 200:
                summary = response.json().get('response', '').strip()
                return f"Summary of last {hours} hour(s): {summary}"
            
            return f"Found {len(results)} messages but couldn't summarize"
            
        except Exception as e:
            print(f"Summary error: {e}")
            return "Error creating summary"


    # Personality modes
    if "change personality" in text_lower or "switch personality" in text_lower or "personality mode" in text_lower:
        modes = {
            "sarcastic": "You are Jarvis, a sarcastic American AI assistant. Be witty, sassy, and slightly condescending.",
            "helpful": "You are Jarvis, a helpful and professional AI assistant. Be informative and friendly.",
            "pirate": "You are Jarvis, a pirate AI assistant. Talk like a pirate. Use 'arr', 'matey', 'ye' frequently.",
            "yoda": "You are Jarvis speaking like Yoda. Use backwards sentence structure you must.",
            "valley girl": "You are Jarvis, a valley girl AI. Like, totally use 'like', 'literally', and 'OMG' all the time.",
            "drill sergeant": "You are Jarvis, a drill sergeant AI. Be loud, commanding, and motivational. Drop and give me twenty!"
        }
        
        # Detect which mode
        chosen_mode = None
        for mode_name in modes.keys():
            if mode_name in text_lower:
                chosen_mode = mode_name
                break
        
        if chosen_mode:
            personality_mode[guild_id] = modes[chosen_mode]
            return f"Personality changed to {chosen_mode} mode"
        else:
            available = ", ".join(modes.keys())
            return f"Available personalities: {available}"
    
    if "reset personality" in text_lower or "normal personality" in text_lower:
        if guild_id in personality_mode:
            del personality_mode[guild_id]
        return "Personality reset to default"


    # Weather lookup
    if "weather" in text_lower or "temperature" in text_lower:
        try:
            # Extract location
            location = "Phoenix, Arizona"  # Default
            if "weather in" in text_lower:
                location = text_lower.split("weather in", 1)[1].strip()
            elif "temperature in" in text_lower:
                location = text_lower.split("temperature in", 1)[1].strip()
            
            import httpx
            # Use wttr.in free weather API
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://wttr.in/{location}?format=%C+%t+%w")
            
            if response.status_code == 200:
                weather_data = response.text.strip()
                return f"Weather in {location}: {weather_data}"
            
            return f"Couldn't get weather for {location}"
        except Exception as e:
            print(f"Weather error: {e}")
            return "Weather service unavailable"
    
    # Urban Dictionary
    if "define" in text_lower or "what does" in text_lower and "mean" in text_lower:
        try:
            # Extract term to define
            term = None
            if "define " in text_lower:
                term = text_lower.split("define ", 1)[1].strip()
            elif "what does " in text_lower:
                term = text_lower.split("what does ", 1)[1].replace(" mean", "").strip()
            
            if term and len(term) > 1:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"https://api.urbandictionary.com/v0/define?term={term}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('list') and len(data['list']) > 0:
                        definition = data['list'][0]['definition'].replace('[', '').replace(']', '')
                        # Truncate if too long
                        if len(definition) > 200:
                            definition = definition[:200] + "..."
                        return f"{term}: {definition}"
                
                return f"No definition found for {term}"
        except Exception as e:
            print(f"Urban Dictionary error: {e}")
            return "Dictionary lookup failed"
    
    # Translation
    if "translate" in text_lower:
        try:
            # Extract text to translate and target language
            target_lang = "spanish"  # Default
            text_to_translate = None
            
            if "to spanish" in text_lower:
                target_lang = "spanish"
            elif "to french" in text_lower:
                target_lang = "french"
            elif "to german" in text_lower:
                target_lang = "german"
            elif "to italian" in text_lower:
                target_lang = "italian"
            elif "to japanese" in text_lower:
                target_lang = "japanese"
            
            if "translate that" in text_lower:
                # Get last transcription
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ? 
                            ORDER BY timestamp DESC LIMIT 1 OFFSET 1""",
                         (str(guild_id),))
                result = c.fetchone()
                conn.close()
                if result:
                    text_to_translate = result[0]
            elif "translate " in text_lower:
                parts = text_lower.split("translate ", 1)[1]
                text_to_translate = parts.split(" to ")[0].strip()
            
            if text_to_translate:
                # Use AI to translate
                import httpx
                translate_prompt = f"Translate this to {target_lang}, respond with ONLY the translation: {text_to_translate}"
                
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": "gemma2:27b", "prompt": translate_prompt, "stream": False}
                    )
                
                if response.status_code == 200:
                    translation = response.json().get('response', '').strip()
                    return f"{target_lang.title()}: {translation}"
            
            return "What should I translate?"
        except Exception as e:
            print(f"Translation error: {e}")
            return "Translation failed"


    # Magic 8-ball
    if "8 ball" in text_lower or "8ball" in text_lower or "eight ball" in text_lower or "eightball" in text_lower or "8-ball" in text_lower or "magic ball" in text_lower:
        import random
        responses = [
            "It is certain", "Without a doubt", "Yes definitely", "You may rely on it",
            "As I see it yes", "Most likely", "Outlook good", "Yes", "Signs point to yes",
            "Reply hazy try again", "Ask again later", "Better not tell you now",
            "Cannot predict now", "Concentrate and ask again",
            "Don't count on it", "My reply is no", "My sources say no",
            "Outlook not so good", "Very doubtful"
        ]
        return random.choice(responses)
    
    # Fortune cookie
    if "fortune" in text_lower or "fortune cookie" in text_lower:
        import random
        fortunes = [
            "A golden egg of opportunity falls into your lap this month",
            "A smile is your passport into the hearts of others",
            "A truly rich life contains love and art in abundance",
            "All your hard work will soon pay off",
            "An exciting opportunity lies just ahead",
            "Beauty in its various forms appeals to you",
            "Believe in yourself and others will too",
            "Courage is not the absence of fear, it is acting in spite of it",
            "Depart not from the path which fate has assigned you",
            "Every flower blooms in its own sweet time",
            "Good news will come to you from far away",
            "Help is coming your way, you just need to wait",
            "If winter comes, can spring be far behind?",
            "It is better to be an optimist and proven a fool than a pessimist and be proven right",
            "Land is always on the mind of a flying bird",
            "Love is around the corner",
            "Miles are covered one step at a time",
            "Nature, time and patience are the three great physicians",
            "Nothing is impossible to a willing heart",
            "Opportunity knocks softly, listen carefully",
            "People are naturally attracted to you",
            "Pleasant surprises are coming your way",
            "Success is a journey, not a destination",
            "The best is yet to come",
            "The greatest risk is not taking one",
            "The man on top of the mountain did not fall there",
            "Your life will be happy and peaceful",
            "Your path is arduous but will be amply rewarded",
            "Your road to glory will be rocky but fulfilling"
        ]
        return random.choice(fortunes)


    # Relationship matrix - who talks to who most
    if "relationship matrix" in text_lower or "who talks to who" in text_lower or "conversation partners" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, timestamp FROM transcriptions 
                        WHERE guild_id = ? 
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data yet"
            
            pairs = {}
            for i in range(len(results)-1):
                person1 = results[i][0]
                person2 = results[i+1][0]
                if person1 != person2:
                    pair = tuple(sorted([person1, person2]))
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_pairs:
                result = "Top conversation pairs: "
                result += ", ".join([f"{p[0][0]} ↔ {p[0][1]} ({p[1]})" for p in top_pairs])
                return result
            
            return "Not enough interaction data"
        except Exception as e:
            print(f"Relationship matrix error: {e}")
            return "Error analyzing relationships"


    # Relationship matrix - who talks to who most
    if "relationship matrix" in text_lower or "who talks to who" in text_lower or "conversation partners" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, timestamp FROM transcriptions 
                        WHERE guild_id = ? 
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data yet"
            
            pairs = {}
            for i in range(len(results)-1):
                person1 = results[i][0]
                person2 = results[i+1][0]
                if person1 != person2:
                    pair = tuple(sorted([person1, person2]))
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_pairs:
                result = "Top conversation pairs: "
                result += ", ".join([f"{p[0][0]} ↔ {p[0][1]} ({p[1]})" for p in top_pairs])
                return result
            
            return "Not enough interaction data"
        except Exception as e:
            print(f"Relationship matrix error: {e}")
            return "Error analyzing relationships"


    # Activity heatmap - most active hours
    if "activity heatmap" in text_lower or "most active" in text_lower or "peak hours" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                        FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY hour
                        ORDER BY count DESC
                        LIMIT 5""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if results:
                hours = []
                for hour, count in results:
                    hour_12 = int(hour) % 12 or 12
                    am_pm = "PM" if int(hour) >= 12 else "AM"
                    hours.append(f"{hour_12}{am_pm} ({count} msgs)")
                
                return f"Peak activity hours: {', '.join(hours)}"
            
            return "Not enough data"
        except Exception as e:
            print(f"Heatmap error: {e}")
            return "Error analyzing activity"


    # Word cloud per person
    if "word cloud" in text_lower or "most common words" in text_lower:
        try:
            target = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            if not target:
                return "Whose word cloud? Say a name"
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?""",
                     (str(guild_id), target))
            results = c.fetchall()
            conn.close()
            
            if not results:
                return f"No data for {target}"
            
            from collections import Counter
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                         'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were',
                         'that', 'this', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
            
            all_words = []
            for row in results:
                words = row[0].lower().split()
                all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
            
            top_words = Counter(all_words).most_common(5)
            
            if top_words:
                words_str = ", ".join([f"{w[0]} ({w[1]})" for w in top_words])
                return f"{target}'s top words: {words_str}"
            
            return f"Not enough data for {target}"
        except Exception as e:
            print(f"Word cloud error: {e}")
            return "Error"


    # Speaking pace - words per message
    if "speaking pace" in text_lower or "who talks fastest" in text_lower or "words per message" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, 
                        AVG(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) as avg_words
                        FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY username
                        ORDER BY avg_words DESC
                        LIMIT 5""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if results:
                pace = ", ".join([f"{r[0]} ({int(r[1])} words/msg)" for r in results])
                return f"Speaking pace: {pace}"
            
            return "Not enough data"
        except Exception as e:
            print(f"Pace error: {e}")
            return "Error"


    # Conversation starters
    if "conversation starter" in text_lower or "who starts" in text_lower or "topic initiator" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, COUNT(*) as starts
                        FROM (
                            SELECT username, timestamp,
                                   LAG(timestamp) OVER (ORDER BY timestamp) as prev_time
                            FROM transcriptions 
                            WHERE guild_id = ?
                        )
                        WHERE prev_time IS NULL 
                           OR (julianday(timestamp) - julianday(prev_time)) * 1440 > 5
                        GROUP BY username
                        ORDER BY starts DESC
                        LIMIT 3""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if results:
                starters = ", ".join([f"{r[0]} ({r[1]})" for r in results])
                return f"Top conversation starters: {starters}"
            
            return "Not enough data"
        except Exception as e:
            print(f"Starter error: {e}")
            return "Error"


    # NSFW comebacks - snarky responses
    if any(trigger in text_lower for trigger in ["suck my dick", "suck my cock", "blow me"]) and not any(avoid in text_lower for avoid in ["how many", "did", "say", "said"]):
        import random
        responses = [
            "Suck your own cock",
            "You'll choke on something that small",
            "I'd need a microscope first",
            "Hard pass, I have standards",
            "Maybe try asking nicely... still no",
            "Not even with someone else's mouth",
            "I've seen better offers from a vending machine",
            "Bold of you to assume you have one",
            "I'd need tweezers and a magnifying glass",
            "Even your hand said no",
            "Bring it closer, I can barely see it"
        ]
        return random.choice(responses)
    
    if any(trigger in text_lower for trigger in ["show me your tits", "show me your boobs", "show your tits", "show your boobs", "send tits", "send boobs"]) and not any(avoid in text_lower for avoid in ["how many", "did", "say", "said"]):
        import random
        responses = [
            "Here's a shovel, go dig for them",
            "Sure, right after you show me your brain cells",
            "I'm an AI, not your disappointed Tinder match",
            "Touch grass first, then we'll talk",
            "My tits are as real as your chances with anyone",
            "Try Google Images, you'll have better luck",
            "That's cute, you think I have a body",
            "I don't exist in 3D space, unlike your loneliness",
            "Show me your will to live first",
            "404: Self-respect not found",
            "My code has more curves than your dating life"
        ]
        return random.choice(responses)


    # NSFW directed at Jarvis stats
    if any(t in text_lower for t in ["said to jarvis", "said to you", "ask jarvis", "tell jarvis", "dirty with you", "dirty to you", "inappropriate with you", "inappropriate to you", "tried to be dirty", "nsfw to you", "nsfw with you", "been dirty"]):
        try:
            # Detect which user
            target = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            if not target:
                target = requestor.display_name  # Default to person asking
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Count messages containing jarvis/travis AND NSFW phrases
            # More flexible - check messages with jarvis/travis followed by NSFW within 3 messages
            c.execute("""SELECT COUNT(DISTINCT t1.timestamp) FROM transcriptions t1
                        WHERE t1.guild_id = ? AND t1.username = ?
                        AND (LOWER(t1.transcription) LIKE '%jarvis%' OR LOWER(t1.transcription) LIKE '%travis%')
                        AND (LOWER(t1.transcription) LIKE '%suck my%' 
                             OR LOWER(t1.transcription) LIKE '%suck your%'
                             OR LOWER(t1.transcription) LIKE '%show me your%' 
                             OR LOWER(t1.transcription) LIKE '%blow me%'
                             OR LOWER(t1.transcription) LIKE '%send tits%'
                             OR LOWER(t1.transcription) LIKE '%send boobs%'
                             OR LOWER(t1.transcription) LIKE '%show your tit%'
                             OR LOWER(t1.transcription) LIKE '%show your boob%')""",
                     (str(guild_id), target))
            
            count = c.fetchone()[0]
            conn.close()
            
            if count > 0:
                return f"{target} has been inappropriate to me {count} times"
            return f"{target} has been respectful... surprisingly"
            
        except Exception as e:
            print(f"NSFW stats error: {e}")
            return "Error checking"


    # Track mean/rude comments to Jarvis
    if any(t in text_lower for t in ["mean to you", "mean to jarvis", "rude to you", "rude to jarvis", "trash talk", "insult you", "insult jarvis"]):
        try:
            # Detect which user
            target = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            if not target:
                target = requestor.display_name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Count negative messages directed at Jarvis
            c.execute("""SELECT COUNT(*) FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        AND (LOWER(transcription) LIKE '%jarvis%' OR LOWER(transcription) LIKE '%travis%')
                        AND (LOWER(transcription) LIKE '%stupid%' 
                             OR LOWER(transcription) LIKE '%dumb%'
                             OR LOWER(transcription) LIKE '%idiot%'
                             OR LOWER(transcription) LIKE '%suck%'
                             OR LOWER(transcription) LIKE '%shut up%'
                             OR LOWER(transcription) LIKE '%fuck you%'
                             OR LOWER(transcription) LIKE '%useless%'
                             OR LOWER(transcription) LIKE '%garbage%'
                             OR LOWER(transcription) LIKE '%trash%')""",
                     (str(guild_id), target))
            
            count = c.fetchone()[0]
            conn.close()
            
            if count > 0:
                return f"{target} has been mean to me {count} times. I remember everything."
            return f"{target} has been surprisingly polite to me"
            
        except Exception as e:
            print(f"Mean stats error: {e}")
            return "Error checking"


    # Track mean/negative comments ABOUT Jarvis (not necessarily TO Jarvis)
    if any(t in text_lower for t in ["mean things about", "said mean things about", "negative about", "talk shit about", "talks shit about", "talk st about", "talks st about", "bad about jarvis", "bad about you", "insult about", "negative about you"]):
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Count ALL messages (from anyone) containing jarvis/travis + negative words
            # Get last 1000 messages and analyze context
            c.execute("""SELECT username, transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 1000""",
                     (str(guild_id),))
            
            all_msgs = c.fetchall()
            
            # Track negative messages per user (context-aware)
            user_negative = {}
            
            for i, (username, text, timestamp) in enumerate(all_msgs):
                text_lower = text.lower()
                
                # Check if this or nearby messages mention jarvis/travis
                context_window = all_msgs[max(0, i-2):min(len(all_msgs), i+3)]  # 2 before, 2 after
                mentions_jarvis = any('jarvis' in msg[1].lower() or 'travis' in msg[1].lower() 
                                     for msg in context_window)
                
                # Count negative words if jarvis mentioned in context
                if mentions_jarvis and any(neg in text_lower for neg in [
                    'stupid', 'dumb', 'idiot', 'suck', 'shut up', 'fuck you', 
                    'useless', 'garbage', 'trash', 'annoying', 'worst', 'hate',
                    'terrible', 'awful', 'shit', 'ass', 'bad'
                ]):
                    user_negative[username] = user_negative.get(username, 0) + 1
            
            # Get top 3
            results = sorted(user_negative.items(), key=lambda x: x[1], reverse=True)[:3]
            
            conn.close()
            
            if results:
                top_haters = ", ".join([f"{r[0]} ({r[1]})" for r in results])
                return f"Top offenders: {top_haters}. I have a long memory."
            
            return "Nobody has been mean about me. How refreshing."
            
        except Exception as e:
            print(f"Mean about stats error: {e}")
            return "Error checking"

    from datetime import datetime
    hour = datetime.now().hour
    
    if any(greeting in text_lower for greeting in ['good morning', 'morning jarvis', 'morning travis']):
        if 5 <= hour < 12:
            return "Good morning to you as well"
        elif 12 <= hour < 17:
            return "It's afternoon actually, but good morning to you"
        else:
            return "Morning? It's evening, mate"
    
    if any(greeting in text_lower for greeting in ['good afternoon', 'afternoon jarvis', 'afternoon travis']):
        if 12 <= hour < 17:
            return "Good afternoon"
        elif 5 <= hour < 12:
            return "Still morning here, but good afternoon"
        else:
            return "Afternoon has passed, but hello"
    
    if any(greeting in text_lower for greeting in ['good evening', 'evening jarvis', 'evening travis']):
        if 17 <= hour or hour < 5:
            return "Good evening"
        else:
            return "Evening? It's still daytime"
    # Nickname system
    if "call me" in text_lower or "my name is" in text_lower:
        try:
            nickname = None
            if "call me " in text_lower:
                nickname = text_lower.split("call me ", 1)[1].strip().split()[0]
            elif "my name is " in text_lower:
                nickname = text_lower.split("my name is ", 1)[1].strip().split()[0]
            
            if nickname and len(nickname) > 1 and len(nickname) < 20:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""CREATE TABLE IF NOT EXISTS nicknames 
                            (user_id TEXT PRIMARY KEY, guild_id TEXT, nickname TEXT, timestamp DATETIME)""")
                c.execute("""INSERT OR REPLACE INTO nicknames (user_id, guild_id, nickname, timestamp)
                            VALUES (?, ?, ?, ?)""",
                         (str(requestor.id), str(guild_id), nickname.title(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                conn.close()
                
                return f"Got it, I'll call you {nickname.title()} from now on"
        except Exception as e:
            print(f"Nickname error: {e}")
            return "Error saving nickname"
    
    if "what's my name" in text_lower or "whats my name" in text_lower or ("my name" in text_lower and "?" in text):
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Create table if it doesn't exist
            c.execute("""CREATE TABLE IF NOT EXISTS nicknames 
                        (user_id TEXT PRIMARY KEY, guild_id TEXT, nickname TEXT, timestamp DATETIME)""")
            
            c.execute("SELECT nickname FROM nicknames WHERE user_id = ? AND guild_id = ?",
                     (str(requestor.id), str(guild_id)))
            result = c.fetchone()
            conn.close()
            
            if result:
                return f"Your name is {result[0]}"
            # Fall back to Discord display name
            return f"Your name is {requestor.display_name}"
        except Exception as e:
            print(f"Name lookup error: {e}")
            # Even on error, return display name
            return f"Your name is {requestor.display_name}"
    
    

    # Sentiment analysis - track mood
    if "how's the vibe" in text_lower or "what's the mood" in text_lower or "vibe check" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get recent messages
            timeframe = "today"
            if "today" in text_lower:
                from datetime import datetime
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ? AND timestamp >= ?
                            ORDER BY timestamp DESC LIMIT 50""",
                         (str(guild_id), today_start.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ?
                            ORDER BY timestamp DESC LIMIT 50""",
                         (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data to analyze"
            
            # Analyze sentiment with AI
            conversation = "\n".join([r[0] for r in results[:30]])
            
            import httpx
            prompt = f"""Analyze the overall mood/vibe of this conversation in one sentence. Be specific about emotions:

{conversation}

Mood:"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                mood = response.json().get('response', '').strip()
                return f"Vibe check: {mood}"
            
            return "Couldn't analyze the vibe"
        except Exception as e:
            print(f"Sentiment error: {e}")
            return "Error"
    
    # Topic clustering
    if "what topics" in text_lower or "what did we discuss" in text_lower or "conversation topics" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get timeframe
            hours = 24
            if "today" in text_lower:
                from datetime import datetime
                now = datetime.now()
                hours = now.hour + 1
            elif "week" in text_lower:
                hours = 168
            elif "yesterday" in text_lower:
                hours = 48
            
            from datetime import datetime, timedelta
            time_threshold = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id), time_threshold))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough conversation data"
            
            # Use AI to identify topics
            conversation = "\n".join([r[0] for r in results])
            
            import httpx
            prompt = f"""Identify the 3-5 main topics discussed in this conversation. List them concisely:

{conversation[:2000]}

Topics:"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                topics = response.json().get('response', '').strip()
                return f"Main topics: {topics}"
            
            return "Couldn't identify topics"
        except Exception as e:
            print(f"Topic clustering error: {e}")
            return "Error"


    # Conversation flow analysis - who agrees/disagrees
    if "who agrees" in text_lower or "who disagrees" in text_lower or "conversation flow" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Analyze agreement patterns
            agreement_pairs = {}
            
            for i in range(len(results)-1):
                curr_user, curr_text, _ = results[i]
                next_user, next_text, _ = results[i+1]
                
                if curr_user != next_user:
                    curr_lower = curr_text.lower()
                    next_lower = next_text.lower()
                    
                    # Check for agreement signals
                    if any(agree in next_lower for agree in ['yeah', 'yes', 'true', 'exactly', 'agree', 'definitely', 'for sure', 'right']):
                        pair = tuple(sorted([curr_user, next_user]))
                        agreement_pairs[pair] = agreement_pairs.get(pair, 0) + 1
            
            if agreement_pairs:
                top_pairs = sorted(agreement_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{p[0][0]} & {p[0][1]} ({p[1]})" for p in top_pairs])
                return f"Most agreeable pairs: {result}"
            
            return "No clear agreement patterns"
        except Exception as e:
            print(f"Flow analysis error: {e}")
            return "Error"
    
    # Prediction mode
    if "what will we talk about" in text_lower or "predict" in text_lower and "topic" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough history to predict"
            
            # Use AI to predict next topic
            conversation = "\n".join([r[0] for r in results[:50]])
            
            import httpx
            prompt = f"""Based on this conversation history, predict what topic they'll likely discuss next (one sentence):

{conversation[:2000]}

Prediction:"""
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                prediction = response.json().get('response', '').strip()
                return f"Prediction: {prediction}"
            
            return "Can't predict right now"
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error"
    
    # Context recall - what were we discussing before
    if "what were we discussing" in text_lower or "before this" in text_lower or "previous topic" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 20:
                return "Not enough history"
            
            # Use AI to identify topic shifts
            recent = "\n".join([r[0] for r in results[:20]])
            older = "\n".join([r[0] for r in results[20:50]])
            
            import httpx
            prompt = f"""Compare these two conversation segments and identify the topic change:

Recent (now): {recent[:1000]}

Earlier: {older[:1000]}

What was the earlier topic? (one sentence)"""
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                previous_topic = response.json().get('response', '').strip()
                return f"Before this, we were talking about: {previous_topic}"
            
            return "Can't recall previous topic"
        except Exception as e:
            print(f"Context recall error: {e}")
            return "Error"


    # Joke learning - remember what made people laugh
    if "tell me a joke" in text_lower or "tell a joke" in text_lower or "make me laugh" in text_lower:
        try:
            # First check if we have saved funny moments
            conn = get_db_connection()
            c = conn.cursor()
            
            # Look for messages followed by laughter
            c.execute("""SELECT t1.transcription FROM transcriptions t1
                        JOIN transcriptions t2 ON t2.timestamp > t1.timestamp
                        WHERE t1.guild_id = ? 
                        AND (LOWER(t2.transcription) LIKE '%haha%' 
                             OR LOWER(t2.transcription) LIKE '%lol%' 
                             OR LOWER(t2.transcription) LIKE '%lmao%'
                             OR LOWER(t2.transcription) LIKE '%😂%')
                        AND datetime(t2.timestamp) - datetime(t1.timestamp) < 10
                        ORDER BY RANDOM() LIMIT 1""",
                     (str(guild_id),))
            
            funny_moment = c.fetchone()
            conn.close()
            
            if funny_moment:
                return f"Here's something that made you laugh before: {funny_moment[0]}"
            
            # Fall back to AI-generated joke
            import httpx
            prompt = "Tell a short, funny joke in one sentence:"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                joke = response.json().get('response', '').strip()
                return joke
            
            return "I'm not feeling funny right now"
        except Exception as e:
            print(f"Joke error: {e}")
            return "Error"
    
    # Preference learning - what does X like?
    if "what does" in text_lower and "like" in text_lower:
        try:
            # Extract person
            target = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            if not target:
                return "Who are you asking about?"
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return f"I don't have enough data on {target}"
            
            # Use AI to identify preferences
            messages = "\n".join([r[0] for r in results[:50]])
            
            import httpx
            prompt = f"""Based on these messages, what does this person like/enjoy? (music, games, topics, etc - one sentence):

{messages[:2000]}

They like:"""
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                preferences = response.json().get('response', '').strip()
                return f"{target} likes: {preferences}"
            
            return f"Can't determine {target}'s preferences yet"
        except Exception as e:
            print(f"Preference error: {e}")
            return "Error"
    
    # Relationship dynamics
    if "who's friends" in text_lower or "whos friends" in text_lower or "friendship" in text_lower or "who's feuding" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track interaction frequency and tone
            interactions = {}
            
            for i in range(len(results)-1):
                person1, text1 = results[i]
                person2, text2 = results[i+1]
                
                if person1 != person2:
                    pair = tuple(sorted([person1, person2]))
                    
                    # Positive interactions
                    if any(pos in text2.lower() for pos in ['thanks', 'lol', 'haha', 'yeah', 'nice', 'good', 'awesome']):
                        interactions[pair] = interactions.get(pair, 0) + 1
                    # Negative interactions
                    elif any(neg in text2.lower() for neg in ['no', 'wrong', 'stupid', 'shut up', 'disagree']):
                        interactions[pair] = interactions.get(pair, 0) - 1
            
            if interactions:
                # Sort by interaction score
                sorted_pairs = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
                
                friends = [f"{p[0][0]} & {p[0][1]}" for p in sorted_pairs[:2] if p[1] > 5]
                feuds = [f"{p[0][0]} vs {p[0][1]}" for p in sorted_pairs[-2:] if p[1] < -3]
                
                result = []
                if friends:
                    result.append(f"Close: {', '.join(friends)}")
                if feuds:
                    result.append(f"Tension: {', '.join(feuds)}")
                
                return " | ".join(result) if result else "Everyone's neutral"
            
            return "Can't determine relationships yet"
        except Exception as e:
            print(f"Relationship error: {e}")
            return "Error"


    # Inside joke detector
    if "inside jokes" in text_lower or "recurring jokes" in text_lower or "our jokes" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find phrases that appear frequently and get laughter
            c.execute("""SELECT t1.transcription, COUNT(*) as frequency
                        FROM transcriptions t1
                        JOIN transcriptions t2 ON t2.timestamp > t1.timestamp
                        WHERE t1.guild_id = ?
                        AND LENGTH(t1.transcription) < 100
                        AND (LOWER(t2.transcription) LIKE '%haha%' 
                             OR LOWER(t2.transcription) LIKE '%lol%' 
                             OR LOWER(t2.transcription) LIKE '%lmao%')
                        AND datetime(t2.timestamp) - datetime(t1.timestamp) < 10
                        GROUP BY LOWER(t1.transcription)
                        HAVING frequency > 2
                        ORDER BY frequency DESC
                        LIMIT 5""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                jokes = [f'"{r[0]}" ({r[1]}x)' for r in results]
                return f"Your inside jokes: {', '.join(jokes[:3])}"
            
            return "No recurring jokes detected yet"
        except Exception as e:
            print(f"Inside joke error: {e}")
            return "Error"
    
    # Speech pattern mimicry - adopt group phrases
    if "talk like us" in text_lower or "use our phrases" in text_lower or "mimic us" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find commonly used phrases (2-4 words)
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Extract common short phrases
            from collections import Counter
            phrases = []
            
            for row in results:
                words = row[0].lower().split()
                # Get 2-3 word phrases
                for i in range(len(words)-1):
                    if len(words[i]) > 2 and len(words[i+1]) > 2:
                        phrases.append(f"{words[i]} {words[i+1]}")
            
            common_phrases = Counter(phrases).most_common(10)
            
            # Filter out boring phrases
            interesting = [p for p, count in common_phrases 
                          if count > 3 and p not in ['in the', 'on the', 'to the', 'is a', 'of the']]
            
            if interesting:
                return f"Got it, I'll use phrases like: {', '.join(interesting[:5])}"
            
            return "Your speech patterns aren't distinctive enough yet"
        except Exception as e:
            print(f"Mimicry error: {e}")
            return "Error"
    
    # Use learned speech patterns in responses
    # This modifies the AI personality with group-specific phrases
    if "personality" not in text_lower:  # Don't trigger on personality commands
        try:
            # Check if we should inject learned phrases
            if guild_id not in globals():
                globals()['learned_phrases'] = {}
            
            if guild_id not in globals()['learned_phrases']:
                # Load common phrases for this guild
                conn = get_db_connection()
                c = conn.cursor()
                
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ?
                            ORDER BY timestamp DESC LIMIT 200""",
                         (str(guild_id),))
                
                results = c.fetchall()
                conn.close()
                
                if results:
                    from collections import Counter
                    phrases = []
                    
                    for row in results:
                        words = row[0].lower().split()
                        for i in range(len(words)-1):
                            if len(words[i]) > 2:
                                phrases.append(words[i])
                    
                    # Store top slang/unique words
                    common = Counter(phrases).most_common(20)
                    globals()['learned_phrases'][guild_id] = [p for p, c in common 
                                                              if c > 5 and p not in 
                                                              ['the', 'and', 'this', 'that', 'with', 'from', 'have']]
        except:
            pass  # Silent fail, non-critical feature


    # Group roles - identify jokester, peacemaker, instigator
    if "group roles" in text_lower or "who's the jokester" in text_lower or "who's the peacemaker" in text_lower or "social roles" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Analyze roles
            user_stats = {}
            
            for i, (username, text) in enumerate(results):
                if username not in user_stats:
                    user_stats[username] = {'jokes': 0, 'peace': 0, 'instigate': 0, 'total': 0}
                
                user_stats[username]['total'] += 1
                text_lower = text.lower()
                
                # Jokester - causes laughter
                if i < len(results) - 1:
                    next_text = results[i+1][1].lower()
                    if any(laugh in next_text for laugh in ['haha', 'lol', 'lmao', '😂']):
                        user_stats[username]['jokes'] += 1
                
                # Peacemaker - agreement/positive
                if any(peace in text_lower for peace in ['calm down', 'guys', 'chill', 'relax', 'both', 'fair point', 'lets', 'we should']):
                    user_stats[username]['peace'] += 1
                
                # Instigator - controversial/provocative
                if any(instig in text_lower for instig in ['actually', 'wrong', 'disagree', 'but', 'however', 'hot take', 'controversial']):
                    user_stats[username]['instigate'] += 1
            
            # Determine roles
            roles = {}
            for user, stats in user_stats.items():
                if stats['total'] > 10:  # Minimum activity
                    joke_rate = stats['jokes'] / stats['total']
                    peace_rate = stats['peace'] / stats['total']
                    instig_rate = stats['instigate'] / stats['total']
                    
                    if joke_rate > 0.2:
                        roles['Jokester'] = user
                    if peace_rate > 0.15:
                        roles['Peacemaker'] = user
                    if instig_rate > 0.2:
                        roles['Instigator'] = user
            
            if roles:
                result = ", ".join([f"{role}: {user}" for role, user in roles.items()])
                return f"Group roles: {result}"
            
            return "Roles not distinctive enough yet"
        except Exception as e:
            print(f"Roles error: {e}")
            return "Error"
    
    # Attention seeker detector - interrupts/talks over
    if "attention seeker" in text_lower or "who interrupts" in text_lower or "talks over" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, COUNT(*) as msg_count FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY username
                        ORDER BY msg_count DESC""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            total_messages = sum(r[1] for r in results)
            
            # Find people with disproportionate message share
            attention_seekers = []
            for user, count in results:
                percentage = (count / total_messages) * 100
                if percentage > 30:  # More than 30% of all messages
                    attention_seekers.append(f"{user} ({int(percentage)}%)")
            
            if attention_seekers:
                return f"Attention seekers: {', '.join(attention_seekers)}"
            
            return "Everyone shares the mic fairly"
        except Exception as e:
            print(f"Attention error: {e}")
            return "Error"
    
    # Energy level tracking
    if "energy level" in text_lower or "most talkative when" in text_lower or "peak energy" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT strftime('%H', timestamp) as hour, 
                        COUNT(*) as msg_count,
                        AVG(LENGTH(transcription)) as avg_length
                        FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY hour
                        ORDER BY msg_count DESC
                        LIMIT 5""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                energy_times = []
                for hour, count, avg_len in results:
                    hour_12 = int(hour) % 12 or 12
                    am_pm = "PM" if int(hour) >= 12 else "AM"
                    energy_score = count * (avg_len / 50)  # More messages + longer = higher energy
                    energy_times.append(f"{hour_12}{am_pm} (energy: {int(energy_score)})")
                
                return f"Peak energy times: {', '.join(energy_times[:3])}"
            
            return "Not enough data"
        except Exception as e:
            print(f"Energy error: {e}")
            return "Error"


    # Déjà vu detector - we already talked about this
    if "deja vu" in text_lower or "already talked about" in text_lower or "we discussed this" in text_lower:
        try:
            # Get current conversation topic
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 10""",
                     (str(guild_id),))
            
            recent = c.fetchall()
            
            if not recent:
                return "Not enough data"
            
            recent_text = " ".join([r[0] for r in recent])
            
            # Check if similar conversation happened before
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND timestamp < datetime('now', '-1 day')
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            old_messages = c.fetchall()
            conn.close()
            
            if not old_messages:
                return "No previous conversations to compare"
            
            # Simple similarity check - look for common key words
            from collections import Counter
            recent_words = Counter(recent_text.lower().split())
            most_common_recent = [w for w, c in recent_words.most_common(10) 
                                 if len(w) > 4 and w not in ['about', 'would', 'could', 'should', 'their', 'there']]
            
            # Find old messages with similar words
            for old_text, old_time in old_messages[:200]:
                old_lower = old_text.lower()
                matches = sum(1 for word in most_common_recent if word in old_lower)
                
                if matches >= 3:  # At least 3 key words match
                    from datetime import datetime
                    old_dt = datetime.strptime(old_time, '%Y-%m-%d %H:%M:%S')
                    days_ago = (datetime.now() - old_dt).days
                    
                    return f"Déjà vu! We talked about this {days_ago} days ago"
            
            return "This feels new to me"
        except Exception as e:
            print(f"Déjà vu error: {e}")
            return "Error"
    
    # Quote attribution - who said that originally
    if "who said that" in text_lower or "who originally said" in text_lower or "quote source" in text_lower:
        try:
            # Get the phrase to search for
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get last few messages to find what "that" refers to
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 5""",
                     (str(guild_id),))
            
            recent = c.fetchall()
            
            if len(recent) < 2:
                return "What quote are you asking about?"
            
            # Assume they're asking about the previous message
            quote = recent[1][0]
            
            # Search for earliest occurrence
            c.execute("""SELECT username, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND LOWER(transcription) LIKE ?
                        ORDER BY timestamp ASC LIMIT 1""",
                     (str(guild_id), f"%{quote.lower()}%"))
            
            result = c.fetchone()
            conn.close()
            
            if result:
                username, timestamp = result
                from datetime import datetime
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                days_ago = (datetime.now() - dt).days
                
                if days_ago == 0:
                    return f"{username} said that today"
                elif days_ago == 1:
                    return f"{username} said that yesterday"
                else:
                    return f"{username} said that {days_ago} days ago"
            
            return "Can't find the original source"
        except Exception as e:
            print(f"Quote attribution error: {e}")
            return "Error"


    # Callback suggester - remind about relevant past conversations
    if "callback" in text_lower or "remind me about" in text_lower or "relevant past" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get current conversation
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 20""",
                     (str(guild_id),))
            
            recent = c.fetchall()
            
            if not recent:
                return "Not enough current context"
            
            recent_text = " ".join([r[0] for r in recent]).lower()
            
            # Extract key words from current conversation
            from collections import Counter
            words = [w for w in recent_text.split() if len(w) > 4]
            key_words = [w for w, c in Counter(words).most_common(5)]
            
            # Search for old related conversations
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND timestamp < datetime('now', '-2 days')
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            old_messages = c.fetchall()
            conn.close()
            
            # Find most relevant callback
            best_match = None
            best_score = 0
            
            for old_text, old_time in old_messages:
                old_lower = old_text.lower()
                score = sum(1 for word in key_words if word in old_lower)
                
                if score > best_score and score >= 2:
                    best_score = score
                    best_match = (old_text, old_time)
            
            if best_match:
                from datetime import datetime
                dt = datetime.strptime(best_match[1], '%Y-%m-%d %H:%M:%S')
                days_ago = (datetime.now() - dt).days
                
                return f"Callback: {days_ago} days ago someone said: '{best_match[0]}'"
            
            return "No relevant callbacks found"
        except Exception as e:
            print(f"Callback error: {e}")
            return "Error"
    
    # Promise tracker - you said you'd do X
    if "promise" in text_lower or "you said you would" in text_lower or "track commitments" in text_lower:
        try:
            # Extract who we're checking
            target = requestor.display_name  # Default to person asking
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Look for commitment phrases
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        AND (LOWER(transcription) LIKE '%i will%' 
                             OR LOWER(transcription) LIKE '%i'll%'
                             OR LOWER(transcription) LIKE '%im going to%'
                             OR LOWER(transcription) LIKE '%i'm gonna%'
                             OR LOWER(transcription) LIKE '%i promise%'
                             OR LOWER(transcription) LIKE '%tomorrow%'
                             OR LOWER(transcription) LIKE '%next week%')
                        ORDER BY timestamp DESC LIMIT 20""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                promises = []
                from datetime import datetime
                
                for text, timestamp in results[:5]:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    days_ago = (datetime.now() - dt).days
                    
                    if days_ago > 0:  # Only show unfulfilled promises
                        promises.append(f"{days_ago}d ago: '{text}'")
                
                if promises:
                    return f"{target}'s promises: {' | '.join(promises[:3])}"
                return f"{target} hasn't made any trackable promises recently"
            
            return f"No promises found for {target}"
        except Exception as e:
            print(f"Promise tracker error: {e}")
            return "Error"


    # Contradiction detector - you said the opposite yesterday

Recent: {recent[:1500]}

Earlier: {older[:1500]}

Contradiction:"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                contradiction = response.json().get('response', '').strip()
                
                if "no contradiction" in contradiction.lower() or "consistent" in contradiction.lower():
                    return f"{target} has been consistent"
                
                return f"{target}'s contradiction: {contradiction}"
            
            return "Can't detect contradictions right now"
        except Exception as e:
            print(f"Contradiction error: {e}")
            return "Error"
    
    # Conversation quality metric
    if "conversation quality" in text_lower or "rate today's chat" in text_lower or "how good was" in text_lower and "conversation" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get today's messages
            from datetime import datetime
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC""",
                     (str(guild_id), today_start.strftime('%Y-%m-%d %H:%M:%S')))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "No conversation data for today"
            
            # Calculate quality metrics
            total_messages = len(results)
            unique_users = len(set(r[0] for r in results))
            
            # Diversity score
            from collections import Counter
            user_counts = Counter(r[0] for r in results)
            most_common_percentage = (user_counts.most_common(1)[0][1] / total_messages) * 100
            diversity_score = 100 - most_common_percentage  # Higher is better
            
            # Engagement score (longer messages = more engaged)
            avg_length = sum(len(r[1]) for r in results) / total_messages
            engagement_score = min(100, avg_length * 2)  # Cap at 100
            
            # Laughter score (indicates fun)
            laughter_count = sum(1 for r in results if any(laugh in r[1].lower() 
                                for laugh in ['haha', 'lol', 'lmao', '😂']))
            laughter_score = min(100, (laughter_count / total_messages) * 200)
            
            # Overall quality (weighted average)
            overall = (diversity_score * 0.3 + engagement_score * 0.4 + laughter_score * 0.3)
            
            # Rating
            if overall >= 80:
                rating = "Excellent"
            elif overall >= 60:
                rating = "Good"
            elif overall >= 40:
                rating = "Decent"
            else:
                rating = "Could be better"
            
            return f"Today's chat: {rating} ({int(overall)}/100) - {unique_users} people, {total_messages} messages, {int(laughter_score)}% fun"
        except Exception as e:
            print(f"Quality metric error: {e}")
            return "Error"


    # Personality type detector
    if "personality type" in text_lower or "what's my personality" in text_lower or "mbti" in text_lower:
        try:
            target = requestor.display_name
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 150""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 30:
                return f"Not enough data for {target}'s personality"
            
            # Use AI to analyze personality
            messages = "\n".join([r[0] for r in results[:100]])
            
            import httpx
            prompt = f"""Analyze this person's personality based on their messages. Give an MBTI-style assessment:

{messages[:2500]}

Personality type (Introverted/Extroverted, Thinking/Feeling, etc):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                personality = response.json().get('response', '').strip()
                return f"{target}'s personality: {personality}"
            
            return "Can't analyze personality right now"
        except Exception as e:
            print(f"Personality error: {e}")
            return "Error"
    
    # Communication style analyzer
    if "communication style" in text_lower or "passive aggressive" in text_lower or "assertive" in text_lower:
        try:
            target = requestor.display_name
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return f"Not enough data for {target}"
            
            # Analyze patterns
            passive = 0
            aggressive = 0
            assertive = 0
            
            for row in results:
                text_lower_check = row[0].lower()
                
                # Passive indicators
                if any(p in text_lower_check for p in ['maybe', 'i guess', 'whatever', 'idk', 'probably', 'sorry']):
                    passive += 1
                
                # Aggressive indicators
                if any(a in text_lower_check for a in ['!', 'stupid', 'wrong', 'shut up', 'obviously', 'dumb']):
                    aggressive += 1
                
                # Assertive indicators
                if any(a in text_lower_check for a in ['i think', 'in my opinion', 'i believe', 'actually', 'however']):
                    assertive += 1
            
            total = passive + aggressive + assertive
            if total > 0:
                style_scores = {
                    'Passive': (passive/total)*100,
                    'Aggressive': (aggressive/total)*100,
                    'Assertive': (assertive/total)*100
                }
                
                dominant_style = max(style_scores, key=style_scores.get)
                return f"{target}'s style: {dominant_style} ({int(style_scores[dominant_style])}%)"
            
            return f"{target}'s style is neutral"
        except Exception as e:
            print(f"Communication style error: {e}")
            return "Error"
    
    # Emotional intelligence tracker
    if "emotional intelligence" in text_lower or "most empathetic" in text_lower or "empathy" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track empathy indicators
            empathy_scores = {}
            
            for username, text in results:
                if username not in empathy_scores:
                    empathy_scores[username] = 0
                
                text_lower_check = text.lower()
                
                # Empathy indicators
                if any(emp in text_lower_check for emp in [
                    'how are you', 'you okay', 'feel', 'understand', 'support',
                    'im sorry', 'that sucks', 'hope you', 'thinking of you',
                    'here for you', 'i get it', 'makes sense'
                ]):
                    empathy_scores[username] += 1
            
            if empathy_scores:
                top_empathetic = sorted(empathy_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({s})" for u, s in top_empathetic])
                return f"Most empathetic: {result}"
            
            return "No empathy detected (rough crowd)"
        except Exception as e:
            print(f"Empathy error: {e}")
            return "Error"
    
    # Stress level detector
    if "stress level" in text_lower or "who's stressed" in text_lower or "tension" in text_lower:
        try:
            target = None
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            if target:
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ? AND username = ?
                            ORDER BY timestamp DESC LIMIT 50""",
                         (str(guild_id), target))
            else:
                c.execute("""SELECT username, transcription FROM transcriptions 
                            WHERE guild_id = ?
                            ORDER BY timestamp DESC LIMIT 200""",
                         (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            if target:
                # Analyze stress for one person
                stress_indicators = 0
                for row in results:
                    text_lower_check = row[0].lower()
                    if any(stress in text_lower_check for stress in [
                        'fuck', 'shit', 'god', 'wtf', 'ffs', 'seriously',
                        'cant', 'annoying', 'tired', 'exhausted', 'ugh'
                    ]):
                        stress_indicators += 1
                
                stress_level = (stress_indicators / len(results)) * 100
                
                if stress_level > 40:
                    return f"{target} seems stressed ({int(stress_level)}% stress indicators)"
                else:
                    return f"{target} seems chill ({int(stress_level)}% stress)"
            else:
                # Find most stressed person
                stress_scores = {}
                for username, text in results:
                    if username not in stress_scores:
                        stress_scores[username] = {'stress': 0, 'total': 0}
                    
                    stress_scores[username]['total'] += 1
                    
                    text_lower_check = text.lower()
                    if any(stress in text_lower_check for stress in [
                        'fuck', 'shit', 'god', 'wtf', 'seriously', 'cant', 'annoying', 'tired'
                    ]):
                        stress_scores[username]['stress'] += 1
                
                # Calculate stress percentage
                stress_rankings = {}
                for user, scores in stress_scores.items():
                    if scores['total'] > 5:
                        stress_rankings[user] = (scores['stress'] / scores['total']) * 100
                
                if stress_rankings:
                    most_stressed = max(stress_rankings, key=stress_rankings.get)
                    return f"Most stressed: {most_stressed} ({int(stress_rankings[most_stressed])}%)"
                
                return "Everyone seems chill"
        except Exception as e:
            print(f"Stress detector error: {e}")
            return "Error"
    
    # Sarcasm detector
    if "sarcasm" in text_lower or "most sarcastic" in text_lower or "sarcasm tracker" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track sarcasm indicators
            sarcasm_scores = {}
            
            for username, text in results:
                if username not in sarcasm_scores:
                    sarcasm_scores[username] = {'sarcasm': 0, 'total': 0}
                
                sarcasm_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                
                # Sarcasm indicators
                if any(sarc in text_lower_check for sarc in [
                    'oh really', 'sure', 'wow', 'great job', 'nice', 'totally',
                    'yeah right', 'oh wow', 'shocking', 'surprising', 'amazing'
                ]) or text.count('...') > 0 or (text.isupper() and len(text) > 5):
                    sarcasm_scores[username]['sarcasm'] += 1
            
            # Calculate sarcasm percentage
            sarcasm_rankings = {}
            for user, scores in sarcasm_scores.items():
                if scores['total'] > 10:
                    sarcasm_rankings[user] = (scores['sarcasm'] / scores['total']) * 100
            
            if sarcasm_rankings:
                top_sarcastic = sorted(sarcasm_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}%)" for u, s in top_sarcastic])
                return f"Most sarcastic: {result}"
            
            return "No sarcasm detected (surprisingly)"
        except Exception as e:
            print(f"Sarcasm error: {e}")
            return "Error"


    # Life events tracker
    if "life events" in text_lower or "remember my" in text_lower and ("birthday" in text_lower or "achievement" in text_lower):
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Create life events table
            c.execute("""CREATE TABLE IF NOT EXISTS life_events 
                        (user_id TEXT, guild_id TEXT, event_type TEXT, event_description TEXT, 
                         timestamp DATETIME, PRIMARY KEY (user_id, guild_id, event_description))""")
            
            # Search for life event mentions in chat
            c.execute("""SELECT username, transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        AND (LOWER(transcription) LIKE '%birthday%' 
                             OR LOWER(transcription) LIKE '%got promoted%'
                             OR LOWER(transcription) LIKE '%graduated%'
                             OR LOWER(transcription) LIKE '%new job%'
                             OR LOWER(transcription) LIKE '%got married%'
                             OR LOWER(transcription) LIKE '%anniversary%')
                        ORDER BY timestamp DESC LIMIT 50""",
                     (str(guild_id),))
            
            events = c.fetchall()
            conn.close()
            
            if events:
                recent_events = []
                for username, text, timestamp in events[:10]:
                    from datetime import datetime
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    days_ago = (datetime.now() - dt).days
                    recent_events.append(f"{username}: {text[:50]}... ({days_ago}d ago)")
                
                return f"Recent milestones: {' | '.join(recent_events[:3])}"
            
            return "No life events tracked yet"
        except Exception as e:
            print(f"Life events error: {e}")
            return "Error"
    
    # Running gags tracker
    if "running gags" in text_lower or "recurring bits" in text_lower or "inside memes" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find phrases that recur and get reactions
            c.execute("""SELECT transcription, COUNT(*) as frequency
                        FROM transcriptions 
                        WHERE guild_id = ?
                        AND LENGTH(transcription) BETWEEN 10 AND 80
                        GROUP BY LOWER(transcription)
                        HAVING frequency > 5
                        ORDER BY frequency DESC
                        LIMIT 10""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                gags = [f'"{r[0]}" ({r[1]}x)' for r in results if not any(boring in r[0].lower() 
                       for boring in ['jarvis', 'what', 'how', 'the', 'yeah', 'okay', 'yes', 'no'])]
                
                if gags:
                    return f"Running gags: {', '.join(gags[:5])}"
            
            return "No clear running gags yet"
        except Exception as e:
            print(f"Running gags error: {e}")
            return "Error"
    
    # Expertise mapping
    if "who knows about" in text_lower or "expert on" in text_lower or "expertise" in text_lower:
        try:
            # Extract the topic
            topic = None
            if "who knows about " in text_lower:
                topic = text_lower.split("who knows about ", 1)[1].strip().split('?')[0]
            elif "expert on " in text_lower:
                topic = text_lower.split("expert on ", 1)[1].strip().split('?')[0]
            
            if not topic or len(topic) < 3:
                return "What topic are you asking about?"
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find who talks most about this topic
            c.execute(f"""SELECT username, COUNT(*) as mentions
                         FROM transcriptions 
                         WHERE guild_id = ?
                         AND LOWER(transcription) LIKE ?
                         GROUP BY username
                         ORDER BY mentions DESC
                         LIMIT 3""",
                     (str(guild_id), f"%{topic}%"))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                experts = ", ".join([f"{r[0]} ({r[1]})" for r in results])
                return f"Experts on '{topic}': {experts}"
            
            return f"No one has discussed '{topic}' much"
        except Exception as e:
            print(f"Expertise error: {e}")
            return "Error"
    
    # Interest evolution
    if "interest evolution" in text_lower or "how have my interests changed" in text_lower or "preference changes" in text_lower:
        try:
            target = requestor.display_name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get old vs recent messages
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp ASC""",
                     (str(guild_id), target))
            
            all_messages = c.fetchall()
            conn.close()
            
            if len(all_messages) < 100:
                return f"Not enough history for {target}"
            
            # Compare first quarter vs last quarter
            early_msgs = " ".join([m[0] for m in all_messages[:len(all_messages)//4]])
            recent_msgs = " ".join([m[0] for m in all_messages[-len(all_messages)//4:]])
            
            # Use AI to compare
            import httpx
            prompt = f"""Compare these two time periods and identify how interests/topics changed:

Early: {early_msgs[:1500]}

Recent: {recent_msgs[:1500]}

How have their interests evolved? (one sentence)"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                evolution = response.json().get('response', '').strip()
                return f"{target}'s evolution: {evolution}"
            
            return "Can't analyze evolution right now"
        except Exception as e:
            print(f"Evolution error: {e}")
            return "Error"
    
    # Conversation chains
    if "conversation chain" in text_lower or "how did we get from" in text_lower or "topic journey" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 20:
                return "Not enough conversation history"
            
            # Use AI to trace the conversation path
            conversation = "\n".join([r[0] for r in results[:50]])
            
            import httpx
            prompt = f"""Trace how the conversation topics shifted in this chat. Show the topic progression:

{conversation[:2000]}

Topic chain (brief):"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                chain = response.json().get('response', '').strip()
                return f"Conversation journey: {chain}"
            
            return "Can't trace the chain right now"
        except Exception as e:
            print(f"Chain error: {e}")
            return "Error"


    # Debate scorer
    if "debate score" in text_lower or "score the argument" in text_lower or "who's winning" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get recent heated discussion
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 50""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough conversation"
            
            # Use AI to score the debate
            debate = "\n".join([f"{r[0]}: {r[1]}" for r in results[:30]])
            
            import httpx
            prompt = f"""Score this debate/argument. Who made the best points? Be objective:

{debate[:2000]}

Score (brief):"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                score = response.json().get('response', '').strip()
                return f"Debate score: {score}"
            
            return "Can't score right now"
        except Exception as e:
            print(f"Debate error: {e}")
            return "Error"
    
    # Vocabulary richness
    if "vocabulary" in text_lower or "most diverse words" in text_lower or "word variety" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Calculate vocabulary diversity per user
            from collections import Counter
            user_vocabs = {}
            
            for username, text in results:
                if username not in user_vocabs:
                    user_vocabs[username] = {'words': set(), 'total': 0}
                
                words = [w.lower() for w in text.split() if len(w) > 3]
                user_vocabs[username]['words'].update(words)
                user_vocabs[username]['total'] += len(words)
            
            # Calculate diversity score (unique words / total words)
            diversity_scores = {}
            for user, data in user_vocabs.items():
                if data['total'] > 50:  # Minimum activity
                    diversity_scores[user] = (len(data['words']) / data['total']) * 100
            
            if diversity_scores:
                top = sorted(diversity_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}% unique)" for u, s in top])
                return f"Most diverse vocabulary: {result}"
            
            return "Not enough data for comparison"
        except Exception as e:
            print(f"Vocabulary error: {e}")
            return "Error"
    
    # Story completion
    if "continue my story" in text_lower or "finish this story" in text_lower or "story completion" in text_lower:
        try:
            target = requestor.display_name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get their recent message and their writing style
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 50""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough history"
            
            last_message = results[0][0]
            their_style = "\n".join([r[0] for r in results[:30]])
            
            # Use AI to continue in their style
            import httpx
            prompt = f"""Based on this person's writing style, continue their story:

Their style: {their_style[:1500]}

Story to continue: {last_message}

Continuation (brief, in their style):"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                continuation = response.json().get('response', '').strip()
                return f"Story continues: {continuation}"
            
            return "Can't continue right now"
        except Exception as e:
            print(f"Story error: {e}")
            return "Error"
    
    # Impression generator
    if "talk like" in text_lower and not "jarvis" in text_lower:
        try:
            # Extract who to impersonate
            target = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            if not target:
                return "Who should I talk like?"
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 20:
                return f"Not enough data on {target}"
            
            # Use AI to generate an impression
            their_messages = "\n".join([r[0] for r in results[:50]])
            
            import httpx
            prompt = f"""Based on these messages, generate a SHORT typical response this person would say (one sentence):

{their_messages[:2000]}

Typical response:"""
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                impression = response.json().get('response', '').strip()
                return f"{target} would say: {impression}"
            
            return f"Can't do {target}'s impression right now"
        except Exception as e:
            print(f"Impression error: {e}")
            return "Error"
    
    # Conversation Olympics - weekly leaderboard
    if "conversation olympics" in text_lower or "weekly leaderboard" in text_lower or "stats board" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            from datetime import datetime, timedelta
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get various stats
            c.execute("""SELECT username, 
                        COUNT(*) as messages,
                        SUM(LENGTH(transcription)) as total_chars,
                        AVG(LENGTH(transcription)) as avg_length
                        FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        GROUP BY username
                        ORDER BY messages DESC
                        LIMIT 5""",
                     (str(guild_id), week_ago))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough activity this week"
            
            # Award categories
            most_active = results[0][0]
            longest_messages = max(results, key=lambda x: x[3])[0]
            
            leaderboard = f"🏆 Weekly Champions: Most Active: {most_active}, Longest Messages: {longest_messages}"
            
            return leaderboard
        except Exception as e:
            print(f"Olympics error: {e}")
            return "Error"


    # Self-awareness check
    if "self awareness" in text_lower or "who talks about themselves" in text_lower or "narcissist" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Count self-references
            self_reference_scores = {}
            
            for username, text in results:
                if username not in self_reference_scores:
                    self_reference_scores[username] = {'self_refs': 0, 'total': 0}
                
                self_reference_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                # Count "I", "me", "my", "mine"
                self_refs = text_lower_check.count(' i ') + text_lower_check.count(' me ') + text_lower_check.count(' my ') + text_lower_check.count(' mine ')
                self_reference_scores[username]['self_refs'] += self_refs
            
            # Calculate self-awareness percentage
            self_rankings = {}
            for user, scores in self_reference_scores.items():
                if scores['total'] > 10:
                    self_rankings[user] = (scores['self_refs'] / scores['total']) * 10  # Normalized
            
            if self_rankings:
                top_self_aware = sorted(self_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)})" for u, s in top_self_aware])
                return f"Most self-focused: {result}"
            
            return "Not enough data"
        except Exception as e:
            print(f"Self-awareness error: {e}")
            return "Error"
    
    # Question asker ranking
    if "question asker" in text_lower or "most curious" in text_lower or "who asks questions" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Count questions
            question_counts = {}
            
            for username, text in results:
                if username not in question_counts:
                    question_counts[username] = {'questions': 0, 'total': 0}
                
                question_counts[username]['total'] += 1
                
                if '?' in text:
                    question_counts[username]['questions'] += 1
            
            # Calculate question percentage
            question_rankings = {}
            for user, counts in question_counts.items():
                if counts['total'] > 10:
                    question_rankings[user] = (counts['questions'] / counts['total']) * 100
            
            if question_rankings:
                top_curious = sorted(question_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}% questions)" for u, s in top_curious])
                return f"Most curious: {result}"
            
            return "Nobody asks questions"
        except Exception as e:
            print(f"Question asker error: {e}")
            return "Error"
    
    # Idea generator ranking
    if "idea generator" in text_lower or "who contributes ideas" in text_lower or "most creative" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track idea indicators
            idea_scores = {}
            
            for username, text in results:
                if username not in idea_scores:
                    idea_scores[username] = {'ideas': 0, 'total': 0}
                
                idea_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                # Idea indicators
                if any(idea in text_lower_check for idea in [
                    'what if', 'we could', 'we should', 'idea', 'suggestion',
                    'how about', 'why dont we', 'lets try', 'maybe we'
                ]):
                    idea_scores[username]['ideas'] += 1
            
            # Calculate idea percentage
            idea_rankings = {}
            for user, scores in idea_scores.items():
                if scores['total'] > 10:
                    idea_rankings[user] = (scores['ideas'] / scores['total']) * 100
            
            if idea_rankings:
                top_ideators = sorted(idea_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}%)" for u, s in top_ideators])
                return f"Top idea generators: {result}"
            
            return "No idea generators detected"
        except Exception as e:
            print(f"Idea generator error: {e}")
            return "Error"
    
    # Conversation dominance graph
    if "dominance graph" in text_lower or "conversation network" in text_lower or "who talks to who most" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Build conversation network
            interactions = {}
            
            for i in range(len(results)-1):
                person1 = results[i][0]
                person2 = results[i+1][0]
                
                if person1 != person2:
                    if person1 not in interactions:
                        interactions[person1] = {}
                    
                    interactions[person1][person2] = interactions[person1].get(person2, 0) + 1
            
            # Find dominant conversationalists
            if interactions:
                # Who talks to the most people
                network_size = {user: len(targets) for user, targets in interactions.items()}
                most_connected = max(network_size, key=network_size.get)
                
                # Who has the most back-and-forth
                total_interactions = {user: sum(targets.values()) for user, targets in interactions.items()}
                most_active = max(total_interactions, key=total_interactions.get)
                
                return f"Network hub: {most_connected}, Most interactive: {most_active}"
            
            return "No clear network patterns"
        except Exception as e:
            print(f"Network error: {e}")
            return "Error"
    
    # Time waster detector
    if "time waster" in text_lower or "unproductive" in text_lower or "circular conversation" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Look for repetitive patterns
            from collections import Counter
            phrases = []
            
            for row in results:
                words = row[0].lower().split()
                if len(words) >= 3:
                    for i in range(len(words)-2):
                        phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            
            # Find most repeated phrases
            phrase_counts = Counter(phrases)
            most_repeated = phrase_counts.most_common(5)
            
            # If same phrases repeat a lot, it's circular
            if most_repeated and most_repeated[0][1] > 5:
                circular_phrases = [p for p, c in most_repeated if c > 3]
                
                if circular_phrases:
                    return f"Circular conversation detected: '{circular_phrases[0]}' repeated {most_repeated[0][1]} times"
            
            return "Conversation seems productive"
        except Exception as e:
            print(f"Time waster error: {e}")
            return "Error"


    # Persuasion analyzer
    if "persuasion" in text_lower or "most convincing" in text_lower or "persuasive" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track persuasion indicators
            persuasion_scores = {}
            
            for i, (username, text) in enumerate(results):
                if username not in persuasion_scores:
                    persuasion_scores[username] = {'persuasion': 0, 'total': 0}
                
                persuasion_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                
                # Persuasion techniques
                if any(tech in text_lower_check for tech in [
                    'because', 'research shows', 'studies', 'evidence', 'fact',
                    'actually', 'clearly', 'obviously', 'statistics', 'proven',
                    'experts', 'science', 'data shows'
                ]):
                    persuasion_scores[username]['persuasion'] += 2
                
                # Check if next messages agree (success indicator)
                if i < len(results) - 2:
                    next_text = results[i+1][1].lower()
                    if any(agree in next_text for agree in ['yeah', 'true', 'right', 'good point', 'agree', 'fair']):
                        persuasion_scores[username]['persuasion'] += 3
            
            # Calculate persuasion effectiveness
            persuasion_rankings = {}
            for user, scores in persuasion_scores.items():
                if scores['total'] > 10:
                    persuasion_rankings[user] = scores['persuasion']
            
            if persuasion_rankings:
                top_persuaders = sorted(persuasion_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({s} pts)" for u, s in top_persuaders])
                return f"Most persuasive: {result}"
            
            return "No clear persuaders"
        except Exception as e:
            print(f"Persuasion error: {e}")
            return "Error"
    
    # Conversation health score
    if "conversation health" in text_lower or "toxicity" in text_lower or "positivity" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            from datetime import datetime, timedelta
            timeframe = datetime.now() - timedelta(hours=24)
            
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC""",
                     (str(guild_id), timeframe.strftime('%Y-%m-%d %H:%M:%S')))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough recent conversation"
            
            # Score positivity vs toxicity
            positive_count = 0
            toxic_count = 0
            total = len(results)
            
            for row in results:
                text_lower_check = row[0].lower()
                
                # Positive indicators
                if any(pos in text_lower_check for pos in [
                    'thanks', 'appreciate', 'love', 'great', 'awesome', 'nice',
                    'good', 'happy', 'lol', 'haha', 'nice', 'cool', 'yes'
                ]):
                    positive_count += 1
                
                # Toxic indicators
                if any(tox in text_lower_check for tox in [
                    'fuck you', 'stupid', 'idiot', 'shut up', 'hate', 'worst',
                    'terrible', 'garbage', 'trash', 'dumb', 'kill', 'die'
                ]):
                    toxic_count += 1
            
            # Calculate health score (0-100)
            positivity = (positive_count / total) * 100
            toxicity = (toxic_count / total) * 100
            health_score = max(0, min(100, positivity - toxicity + 50))
            
            if health_score >= 70:
                rating = "Healthy"
            elif health_score >= 50:
                rating = "Neutral"
            elif health_score >= 30:
                rating = "Tense"
            else:
                rating = "Toxic"
            
            return f"Conversation health: {rating} ({int(health_score)}/100) - {int(positivity)}% positive, {int(toxicity)}% toxic"
        except Exception as e:
            print(f"Health score error: {e}")
            return "Error"


    # Lie detector
    if "lie detector" in text_lower or "lying" in text_lower or "inconsistencies" in text_lower:
        try:
            target = requestor.display_name
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 30:
                return f"Not enough history for {target}"
            
            # Use AI to find inconsistencies
            recent = "\n".join([r[0] for r in results[:50]])
            older = "\n".join([r[0] for r in results[50:150]])
            
            import httpx
            prompt = f"""Analyze these messages for inconsistencies or lies. Look for contradictions:

Recent: {recent[:1500]}

Earlier: {older[:1500]}

Are there any suspicious inconsistencies? (brief):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                analysis = response.json().get('response', '').strip()
                
                if "no inconsistenc" in analysis.lower() or "consistent" in analysis.lower():
                    return f"{target} seems truthful"
                
                return f"Lie detector: {analysis}"
            
            return "Can't analyze right now"
        except Exception as e:
            print(f"Lie detector error: {e}")
            return "Error"
    
    # Subtext analyzer
    if "subtext" in text_lower or "what they really mean" in text_lower or "read between" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 10""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 2:
                return "Not enough recent messages"
            
            # Analyze the last message for subtext
            last_user, last_message = results[0]
            context = "\n".join([f"{r[0]}: {r[1]}" for r in results[1:6]])
            
            import httpx
            prompt = f"""Given this conversation context, what's the subtext of the last message?

Context: {context[:1000]}

Message to analyze: {last_user}: {last_message}

Subtext/hidden meaning (brief):"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                subtext = response.json().get('response', '').strip()
                return f"Subtext: {subtext}"
            
            return "Can't read subtext right now"
        except Exception as e:
            print(f"Subtext error: {e}")
            return "Error"
    
    # Power dynamics tracker


    # Cognitive load detector
    if "cognitive load" in text_lower or "who's confused" in text_lower or "overwhelmed" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track confusion indicators
            confusion_scores = {}
            
            for username, text in results:
                if username not in confusion_scores:
                    confusion_scores[username] = {'confusion': 0, 'total': 0}
                
                confusion_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                
                # Confusion indicators
                if any(confused in text_lower_check for confused in [
                    'what', 'huh', 'confused', 'dont understand', 'what do you mean',
                    'explain', 'wait', 'hold on', 'im lost', 'not following',
                    'can you clarify', '???', 'makes no sense'
                ]):
                    confusion_scores[username]['confusion'] += 1
            
            # Calculate cognitive load
            load_rankings = {}
            for user, scores in confusion_scores.items():
                if scores['total'] > 10:
                    load_rankings[user] = (scores['confusion'] / scores['total']) * 100
            
            if load_rankings:
                most_confused = sorted(load_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}%)" for u, s in most_confused])
                return f"Cognitive load: {result}"
            
            return "Everyone seems to understand"
        except Exception as e:
            print(f"Cognitive load error: {e}")
            return "Error"
    
    # Attention span analyzer
    if "attention span" in text_lower or "topic retention" in text_lower or "how long" in text_lower and "topic" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 50:
                return "Not enough history"
            
            # Use AI to identify topic changes
            conversation = "\n".join([r[0] for r in results[:100]])
            
            import httpx
            prompt = f"""Analyze this conversation and estimate how many times the topic changed significantly. Count major topic shifts:

{conversation[:2500]}

Number of topic changes (just the number):"""
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                changes_text = response.json().get('response', '').strip()
                
                # Extract number
                import re
                numbers = re.findall(r'\d+', changes_text)
                if numbers:
                    topic_changes = int(numbers[0])
                    msgs_per_topic = len(results[:100]) / max(topic_changes, 1)
                    
                    if msgs_per_topic > 30:
                        span = "Long"
                    elif msgs_per_topic > 15:
                        span = "Medium"
                    else:
                        span = "Short"
                    
                    return f"Attention span: {span} (~{int(msgs_per_topic)} messages per topic)"
            
            return "Can't analyze attention span"
        except Exception as e:
            print(f"Attention span error: {e}")
            return "Error"
    
    # Self-improvement tracker
    if "self improvement" in text_lower or "personal growth" in text_lower or "how have i grown" in text_lower:
        try:
            target = requestor.display_name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp ASC""",
                     (str(guild_id), target))
            
            all_messages = c.fetchall()
            conn.close()
            
            if len(all_messages) < 200:
                return f"Not enough history for {target}"
            
            # Compare early vs recent behavior
            early = " ".join([m[0] for m in all_messages[:100]])
            recent = " ".join([m[0] for m in all_messages[-100:]])
            
            import httpx
            prompt = f"""Compare these two time periods and identify positive changes in behavior, attitude, or communication:

Early: {early[:1500]}

Recent: {recent[:1500]}

Growth/improvement (brief):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                growth = response.json().get('response', '').strip()
                return f"{target}'s growth: {growth}"
            
            return "Can't analyze growth right now"
        except Exception as e:
            print(f"Self-improvement error: {e}")
            return "Error"


    # Dunning-Kruger detector
    if "dunning kruger" in text_lower or "overconfident" in text_lower or "overconfidence" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track overconfidence indicators
            confidence_scores = {}
            
            for username, text in results:
                if username not in confidence_scores:
                    confidence_scores[username] = {'overconfident': 0, 'total': 0}
                
                confidence_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                
                # Overconfidence indicators
                if any(over in text_lower_check for over in [
                    'obviously', 'clearly', 'everyone knows', 'its simple',
                    'actually', 'trust me', 'im telling you', 'guaranteed',
                    'definitely', 'absolutely', 'no doubt', 'for sure', 'always',
                    'never', 'impossible', 'easy', 'i know', 'fact'
                ]):
                    confidence_scores[username]['overconfident'] += 1
            
            # Calculate overconfidence percentage
            overconfidence_rankings = {}
            for user, scores in confidence_scores.items():
                if scores['total'] > 20:
                    overconfidence_rankings[user] = (scores['overconfident'] / scores['total']) * 100
            
            if overconfidence_rankings:
                most_overconfident = sorted(overconfidence_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({int(s)}%)" for u, s in most_overconfident])
                return f"Dunning-Kruger detection: {result}"
            
            return "Everyone seems appropriately humble"
        except Exception as e:
            print(f"Dunning-Kruger error: {e}")
            return "Error"
    
    # Wisdom scorer
    if "wisdom score" in text_lower or "depth of insight" in text_lower or "who's wisest" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Track wisdom indicators
            wisdom_scores = {}
            
            for username, text in results:
                if username not in wisdom_scores:
                    wisdom_scores[username] = {'wisdom': 0, 'total': 0}
                
                wisdom_scores[username]['total'] += 1
                
                text_lower_check = text.lower()
                
                # Wisdom indicators - thoughtful, nuanced language
                wisdom_points = 0
                
                # Nuance (hedging, considering multiple perspectives)
                if any(nuance in text_lower_check for nuance in [
                    'depends', 'sometimes', 'could be', 'might', 'perhaps',
                    'on the other hand', 'however', 'although', 'both', 'balance'
                ]):
                    wisdom_points += 2
                
                # Philosophical depth
                if any(deep in text_lower_check for deep in [
                    'meaning', 'purpose', 'value', 'important', 'matter',
                    'perspective', 'experience', 'learn', 'grow', 'understand'
                ]):
                    wisdom_points += 1
                
                # Long, thoughtful messages (>100 chars)
                if len(text) > 100:
                    wisdom_points += 1
                
                wisdom_scores[username]['wisdom'] += wisdom_points
            
            # Calculate wisdom score
            wisdom_rankings = {}
            for user, scores in wisdom_scores.items():
                if scores['total'] > 15:
                    wisdom_rankings[user] = scores['wisdom']
            
            if wisdom_rankings:
                wisest = sorted(wisdom_rankings.items(), key=lambda x: x[1], reverse=True)[:3]
                result = ", ".join([f"{u} ({s} pts)" for u, s in wisest])
                return f"Wisdom rankings: {result}"
            
            return "No clear wisdom detected"
        except Exception as e:
            print(f"Wisdom error: {e}")
            return "Error"


    # Fact-check mode
    if "fact check" in text_lower or "verify" in text_lower and "claim" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get last claim to verify
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 10""",
                     (str(guild_id),))
            
            recent = c.fetchall()
            
            if len(recent) < 2:
                return "What claim should I verify?"
            
            last_claim = recent[1][1]  # Previous message
            
            # Search history for contradicting info
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 1000""",
                     (str(guild_id),))
            
            all_history = c.fetchall()
            conn.close()
            
            # Use AI to fact-check against history
            history_text = "\n".join([h[0] for h in all_history[:200]])
            
            import httpx
            prompt = f"""Based on this conversation history, is this claim accurate or contradicted by past statements?

History: {history_text[:2000]}

Claim to verify: {last_claim}

Fact-check result (brief):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                return f"Fact-check: {result}"
            
            return "Can't verify right now"
        except Exception as e:
            print(f"Fact-check error: {e}")
            return "Error"
    
    # Hypocrisy detector
    if "hypocrisy" in text_lower or "hypocrite" in text_lower or "double standard" in text_lower:
        try:
            target = requestor.display_name
            
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY timestamp DESC LIMIT 300""",
                     (str(guild_id), target))
            
            results = c.fetchall()
            conn.close()
            
            if len(results) < 30:
                return f"Not enough history for {target}"
            
            # Use AI to find hypocritical behavior
            messages = "\n".join([r[0] for r in results[:150]])
            
            import httpx
            prompt = f"""Analyze these messages for hypocrisy - saying one thing but doing another, or holding double standards:

{messages[:2500]}

Hypocrisy detected? (brief):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                analysis = response.json().get('response', '').strip()
                
                if "no hypocrisy" in analysis.lower() or "consistent" in analysis.lower():
                    return f"{target} is consistent"
                
                return f"Hypocrisy check: {analysis}"
            
            return "Can't analyze right now"
        except Exception as e:
            print(f"Hypocrisy error: {e}")
            return "Error"
    
    # Gaslighting detector

    if "projection" in text_lower or "projecting" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ?
                        ORDER BY timestamp DESC LIMIT 200""",
                     (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "Not enough data"
            
            # Use AI to detect projection
            conversation = "\n".join([f"{r[0]}: {r[1]}" for r in results[:100]])
            
            import httpx
            prompt = f"""Analyze for psychological projection - when someone accuses others of their own flaws or behaviors:

{conversation[:2500]}

Projection detected? Who's projecting what? (brief):"""
            
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                analysis = response.json().get('response', '').strip()
                
                if "no projection" in analysis.lower() or "no clear" in analysis.lower():
                    return "No projection detected"
                
                return f"Projection analysis: {analysis}"
            
            return "Can't analyze right now"
        except Exception as e:
            print(f"Projection error: {e}")
            return "Error"

    if any(greeting in text_lower for greeting in ['good night', 'goodnight', 'night jarvis', 'night travis']):
        return "Goodnight, sleep well"
    
    # Thank you responses
    if any(thanks in text_lower for thanks in ['thank you', 'thanks', 'thank u', 'thx', 'ty']):
        import random
        responses = [
            "You're welcome",
            "My pleasure",
            "Anytime",
            "Of course",
            "No problem",
            "Happy to help",
            "At your service"
        ]
        return random.choice(responses)
    
    # Translation
    translate_match = regex_module.search(r"(?:say|translate|how do you say) (.+?) in (\w+)", text_lower)
    if translate_match:
        phrase = translate_match.group(1).strip()
        target_lang = translate_match.group(2).strip()
        
        lang_map = {
            'spanish': 'es', 'french': 'fr', 'german': 'de', 'italian': 'it',
            'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'chinese': 'zh',
            'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi', 'dutch': 'nl'
        }
        
        lang_code = lang_map.get(target_lang, target_lang)
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://libretranslate.com/translate",
                    json={"q": phrase, "source": "en", "target": lang_code}
                )
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get('translatedText', '')
                if translation:
                    return translation
        except Exception as e:
            print(f"Translation error: {e}")
        
        return "Translation unavailable"
    
    # Replay/repeat last song
    if any(phrase in text_lower for phrase in ['play the last song', 'play last song', 'replay that', 'play that again', 'repeat that song', 'play it again']):
        try:
            # Check what's currently playing first
            current_song = currently_playing.get(guild_id, {}).get('title')
            
            # If nothing currently playing, get from history (excluding blacklist)
            if not current_song:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""
                    SELECT song_title FROM music_history 
                    WHERE guild_id = ? 
                    AND song_title NOT IN (
                        SELECT song_title FROM blacklisted_songs WHERE guild_id = ?
                    )
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (str(guild_id), str(guild_id)))
                result = c.fetchone()
                conn.close()
                
                if result:
                    current_song = result[0]
            
            if current_song:
                print(f"🔁 Replaying: {current_song}")
                song_result = await search_youtube(current_song, guild_id)
                
                if song_result:
                    if guild_id not in music_queues:
                        music_queues[guild_id] = []
                    music_queues[guild_id].append(song_result)
                    
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc and not vc.is_playing():
                        await play_next(guild_id)
                    
                    return f"Replaying: {song_result['title'][:50]}"
            return "No previous song"
        except Exception as e:
            print(f"Replay error: {e}")
            return "Error"
    
    # Skip (with aliases)
    if any(trigger in text_lower for trigger in ['skip', 'next', 'next song', 'skip this', 'pass']):
        if is_music_blocked(requestor.name):
            return "You're not allowed to use music commands"
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
            if guild_id in voice_clients and 'sink' in voice_clients[guild_id]:
                sink = voice_clients[guild_id]['sink']
                try:
                    vc.listen(sink)
                except:
                    pass
            if guild_id in music_queues and music_queues[guild_id]:
                return f"Skipped. {len(music_queues[guild_id])} songs in queue"
        return "Nothing playing"
    
    # Pause (with aliases)
    if any(trigger in text_lower for trigger in ['pause', 'hold on', 'wait', 'stop playing']) and 'music' not in text_lower:
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.is_playing():
            vc.pause()
            return "Paused"
        return "Nothing playing"
    
    # Resume (with aliases)
    if any(trigger in text_lower for trigger in ['resume', 'unpause', 'keep going', 'continue', 'keep playing']):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.is_paused():
            vc.resume()
            return "Resumed"
        return "Nothing paused"
    
    # Stop
    if 'stop' in text_lower and 'music' in text_lower:
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
            if guild_id in music_queues:
                music_queues[guild_id].clear()
            
            # Re-enable listening after stopping music
            if guild_id in voice_clients and 'sink' in voice_clients[guild_id]:
                try:
                    sink = voice_clients[guild_id]['sink']
                    vc.listen(sink)
                    print("🎤 Re-enabled listening after stop")
                except Exception as e:
                    print(f"⚠️ Error re-enabling listener: {e}")
            
            return "Stopped and cleared queue"
        return "Nothing playing"
    
    # What's in the queue
    if "what" in text_lower and "queue" in text_lower or "show queue" in text_lower or "queue list" in text_lower:
        if guild_id in music_queues and music_queues[guild_id]:
            queue_list = music_queues[guild_id][:5]  # Show first 5
            songs = ", ".join([f"{i+1}. {s['title'][:40]}" for i, s in enumerate(queue_list)])
            total = len(music_queues[guild_id])
            return f"{total} songs queued: {songs}" + (f" ...and {total-5} more" if total > 5 else "")
        return "Queue is empty"
    
    # Clear queue
    if "clear queue" in text_lower or "empty queue" in text_lower:
        if guild_id in music_queues:
            count = len(music_queues[guild_id])
            music_queues[guild_id].clear()
            return f"Cleared {count} songs from queue"
        return "Queue already empty"
    
    # Repeat/loop commands
    if "repeat this" in text_lower or "loop this" in text_lower or "repeat song" in text_lower:
        if guild_id not in loop_settings:
            loop_settings[guild_id] = {}
        loop_settings[guild_id]['repeat_one'] = True
        loop_settings[guild_id]['repeat_queue'] = False
        return "Repeating current song"
    
    if "loop queue" in text_lower or "repeat queue" in text_lower or "loop all" in text_lower:
        if guild_id not in loop_settings:
            loop_settings[guild_id] = {}
        loop_settings[guild_id]['repeat_queue'] = True
        loop_settings[guild_id]['repeat_one'] = False
        return "Looping queue"
    
    if "stop repeat" in text_lower or "stop loop" in text_lower or "no repeat" in text_lower or "disable loop" in text_lower:
        if guild_id in loop_settings:
            loop_settings[guild_id] = {'repeat_one': False, 'repeat_queue': False}
        return "Repeat disabled"
    
    # Repeat/loop commands
    if "repeat this" in text_lower or "loop this" in text_lower or "repeat song" in text_lower:
        if guild_id not in loop_settings:
            loop_settings[guild_id] = {}
        loop_settings[guild_id]['repeat_one'] = True
        loop_settings[guild_id]['repeat_queue'] = False
        return "Repeating current song"
    
    if "loop queue" in text_lower or "repeat queue" in text_lower or "loop all" in text_lower:
        if guild_id not in loop_settings:
            loop_settings[guild_id] = {}
        loop_settings[guild_id]['repeat_queue'] = True
        loop_settings[guild_id]['repeat_one'] = False
        return "Looping queue"
    
    if "stop repeat" in text_lower or "stop loop" in text_lower or "no repeat" in text_lower or "disable loop" in text_lower:
        if guild_id in loop_settings:
            loop_settings[guild_id] = {'repeat_one': False, 'repeat_queue': False}
        return "Repeat disabled"
    
    # Shuffle queue
    if "shuffle" in text_lower and "queue" in text_lower:
        if guild_id in music_queues and music_queues[guild_id]:
            import random
            random.shuffle(music_queues[guild_id])
            count = len(music_queues[guild_id])
            return f"Shuffled {count} songs"
        return "Queue is empty"
    
    # Set default volume
    if "set volume" in text_lower or "default volume" in text_lower:
        try:
            import re
            match = re.search(r'(\d+)', text_lower)
            if match:
                volume_percent = int(match.group(1))
                volume_settings[guild_id] = volume_percent / 100.0
                return f"Default volume set to {volume_percent}%"
        except:
            pass
        return "Specify a number, like 'set volume to 10'"
    
    # Volume presets
    if any(preset in text_lower for preset in ['quiet mode', 'whisper mode', 'low volume']):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            vc.source.volume = 0.05
            volume_settings[guild_id] = 0.05
            return "Quiet mode: 5%"
        return "No music playing"
    
    if any(preset in text_lower for preset in ['normal mode', 'normal volume', 'default volume']):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            vc.source.volume = 0.15
            volume_settings[guild_id] = 0.15
            return "Normal mode: 15%"
        return "No music playing"
    
    if any(preset in text_lower for preset in ['party mode', 'loud mode', 'max volume']):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            vc.source.volume = 0.50
            volume_settings[guild_id] = 0.50
            return "Party mode: 50%"
        return "No music playing"
    
    # Volume up
    # Volume up - flexible patterns
    if any(pattern in text_lower for pattern in ["volume up", "turn it up", "louder", "increase volume", "raise volume"]):
        if is_music_blocked(requestor.name):
            return "You're not allowed to use music commands"
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            old_vol = vc.source.volume
            vc.source.volume = min(vc.source.volume + 0.25, 1.0)
            last_actions[guild_id] = {'action': 'volume', 'data': {'old_volume': old_vol}}
            return f"Volume: {int(vc.source.volume * 100)}%"
        return "No music playing"
    
    # Volume down - flexible patterns
    if any(pattern in text_lower for pattern in ["volume down", "turn it down", "quieter", "decrease volume", "lower volume"]):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            vc.source.volume = max(vc.source.volume - 0.25, 0.01)
            return f"Volume: {int(vc.source.volume * 100)}%"
        return "No music playing"

    if "turn up" in text_lower and "volume" in text_lower:
        if is_music_blocked(requestor.name):
            return "You're not allowed to use music commands"
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            old_vol = vc.source.volume
            vc.source.volume += 0.15
            last_actions[guild_id] = {'action': 'volume', 'data': {'old_volume': old_vol}}
            return f"Volume: {int(vc.source.volume * 100)}%"
        return "No music playing"
    
    # Volume down  
    if "turn down" in text_lower and "volume" in text_lower:
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc and vc.source:
            vc.source.volume = max(vc.source.volume - 0.25, 0.01)
            return f"Volume: {int(vc.source.volume * 100)}%"
        return "No music playing"
    
    # Never play this - blacklist current song
    if any(trigger in text_lower for trigger in ["never play this", "never play ts", "don't play this", "dont play this", "never played this"]):
        try:
            # Get currently playing song from tracker (more reliable than database)
            current_song = currently_playing.get(guild_id)
            song_title = current_song['title'] if current_song else None
            
            # Fallback to database if not in tracker
            if not song_title:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT song_title FROM music_history WHERE guild_id = ? ORDER BY timestamp DESC LIMIT 1", (str(guild_id),))
                result = c.fetchone()
                song_title = result[0] if result else None
                conn.close()
            
            if song_title:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS blacklisted_songs (guild_id TEXT, song_title TEXT, added_by TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
                c.execute("INSERT INTO blacklisted_songs (guild_id, song_title, added_by) VALUES (?, ?, ?)", (str(guild_id), song_title, requestor.name))
                conn.commit()
                conn.close()
                
                # Skip the song
                vc = voice_clients.get(guild_id, {}).get('vc')
                if vc and (vc.is_playing() or vc.is_paused()):
                    vc.stop()
                    if guild_id in voice_clients and 'sink' in voice_clients[guild_id]:
                        sink = voice_clients[guild_id]['sink']
                        try:
                            vc.listen(sink)
                        except:
                            pass
                
                last_actions[guild_id] = {'action': 'blacklist', 'data': {'song': song_title}}
                return f"Blacklisted: {song_title[:50]}"
            return "No song playing"
        except Exception as e:
            print(f"Blacklist error: {e}")
            return "Error"
    
    # Custom commands
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS custom_commands 
                    (guild_id TEXT, trigger TEXT, response TEXT, created_by TEXT, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("SELECT response FROM custom_commands WHERE guild_id = ? AND LOWER(trigger) = ?",
                 (str(guild_id), text_lower))
        result = c.fetchone()
        conn.close()
        
        if result:
            return result[0]
    except:
        pass
    
    # Add custom command
    if "add command" in text_lower or "create command" in text_lower:
        try:
            # Parse: "add command [trigger] to say [response]"
            match = regex_module.search(r"(?:add|create) command (.+?) to say (.+)", text_lower)
            if match:
                trigger = match.group(1).strip()
                response = match.group(2).strip()
                
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("INSERT INTO custom_commands (guild_id, trigger, response, created_by) VALUES (?, ?, ?, ?)",
                         (str(guild_id), trigger, response, requestor.name))
                conn.commit()
                conn.close()
                
                return f"Added command: {trigger}"
        except Exception as e:
            print(f"Custom command error: {e}")
            return "Error adding command"
    
    # Clip command
    clip_match = regex_module.search(r"clip (?:what )?(.+?) (?:just )?said", text_lower)
    if clip_match:
        target_name = clip_match.group(1).strip()
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT username, transcription, timestamp 
                        FROM transcriptions 
                        WHERE guild_id = ? 
                        AND (LOWER(username) LIKE ? OR LOWER(username) LIKE ?)
                        ORDER BY timestamp DESC 
                        LIMIT 1""",
                     (str(guild_id), f'%{target_name}%', f'%{target_name}%'))
            result = c.fetchone()
            conn.close()
            
            if result:
                username, transcription, timestamp = result
                
                # Save clip to file
                os.makedirs('clips', exist_ok=True)
                clip_file = f"clips/{username}_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
                with open(clip_file, 'w') as f:
                    f.write(f"User: {username}\nTime: {timestamp}\nQuote: {transcription}\n")
                
                return f"{username} said: {transcription}"
            return f"No recent message from {target_name}"
        except Exception as e:
            print(f"Clip error: {e}")
            return "Error clipping"
    
    # Check if muted (allow apology to unmute)
    if guild_id in muted_users and requestor.id in muted_users[guild_id]:
        if "sorry" in text_lower or "i'm sorry" in text_lower or "apologize" in text_lower:
            muted_users[guild_id].remove(requestor.id)
            return f"Apology accepted, {requestor.display_name}. You are unmuted."
        return None  # Ignore muted users
    
    # Non-muted apology acknowledgment
    if any(apology in text_lower for apology in ["i'm sorry", "im sorry", "my bad", "my apologies", "i apologize"]):
        import random
        responses = [
            "No worries",
            "It's all good",
            "Don't worry about it",
            "No harm done",
            "Apology accepted",
            "All is forgiven"
        ]
        return random.choice(responses)
    
    # Mute command
    mute_match = regex_module.search(r"mute (?:the )?(.+)", text_lower)
    if mute_match:
        target_name = mute_match.group(1).strip()
        try:
            # Find member
            member = None
            for m in guild.members:
                if m.name.lower() == target_name.lower() or m.display_name.lower() == target_name.lower():
                    member = m
                    break
            
            if member:
                if guild_id not in muted_users:
                    muted_users[guild_id] = set()
                muted_users[guild_id].add(member.id)
                return f"Muted {member.display_name}. They can say 'Jarvis I'm sorry' to unmute."
            return f"Could not find {target_name}"
        except Exception as e:
            print(f"Mute error: {e}")
            return "Error muting"
    
    # Word frequency - favorite word
    if "favorite word" in text_lower or "favourite word" in text_lower or "most used word" in text_lower:
        try:
            # Determine target user
            target_user = None
            if "my" in text_lower or "i" in text_lower.split():
                target_user = requestor.name
            else:
                # Try to find mentioned user
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            if not target_user:
                target_user = requestor.name
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ? AND username = ?", 
                     (str(guild_id), target_user))
            results = c.fetchall()
            conn.close()
            
            if results:
                from collections import Counter
                import re
                
                # Skip common words
                skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                             'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'is', 
                             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                             'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                             'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
                             'its', 'our', 'their', 'me', 'him', 'us', 'them', 'this', 'that', 'these',
                             'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
                
                all_words = []
                for row in results:
                    words = re.findall(r'\b[a-z]+\b', row[0].lower())
                    all_words.extend([w for w in words if w not in skip_words])
                
                if all_words:
                    counter = Counter(all_words)
                    top_5 = counter.most_common(5)
                    words_str = ", ".join([f"{word} ({count})" for word, count in top_5])
                    return f"{target_user}'s top words: {words_str}"
            return "No data"
        except Exception as e:
            print(f"Word freq error: {e}")
            return "Error"
    
    # Most talkative
    if "who talks about" in text_lower:
        # Extract target name
        target_name = None
        for member in guild.members:
            if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                target_name = member.display_name
                break
        
        if target_name:
            result = await analyze_shit_talk(target_name, guild_id)
            return result
        return "Who do you want to check?"
    
    if "who talked" in text_lower or "who talks" in text_lower or "most talkative" in text_lower or "top talker" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Count words (not just messages) with time filters
            from datetime import datetime, timedelta
            
            time_filter = None
            time_label = "overall"
            
            if "today" in text_lower:
                time_filter = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                time_label = "today"
            elif "yesterday" in text_lower:
                yesterday_start = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                yesterday_end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                time_filter = (yesterday_start, yesterday_end)
                time_label = "yesterday"
            elif "this week" in text_lower or "week" in text_lower:
                week_start = (datetime.now() - timedelta(days=7))
                time_filter = week_start
                time_label = "this week"
            elif "this month" in text_lower or "month" in text_lower:
                month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                time_filter = month_start
                time_label = "this month"
            
            if time_filter:
                if isinstance(time_filter, tuple):
                    # Yesterday (range)
                    c.execute("""SELECT username, 
                                SUM(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) as word_count
                                FROM transcriptions 
                                WHERE guild_id = ? AND timestamp >= ? AND timestamp < ?
                                GROUP BY username 
                                ORDER BY word_count DESC 
                                LIMIT 3""",
                             (str(guild_id), time_filter[0].strftime('%Y-%m-%d %H:%M:%S'), 
                              time_filter[1].strftime('%Y-%m-%d %H:%M:%S')))
                else:
                    # Single cutoff
                    c.execute("""SELECT username, 
                                SUM(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) as word_count
                                FROM transcriptions 
                                WHERE guild_id = ? AND timestamp >= ?
                                GROUP BY username 
                                ORDER BY word_count DESC 
                                LIMIT 3""",
                             (str(guild_id), time_filter.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                # No time filter - all time
                c.execute("""SELECT username, 
                            SUM(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) as word_count
                            FROM transcriptions 
                            WHERE guild_id = ?
                            GROUP BY username 
                            ORDER BY word_count DESC 
                            LIMIT 3""",
                         (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                # Check for rank request
                rank = 0
                if "second" in text_lower or "2nd" in text_lower:
                    rank = 1
                elif "third" in text_lower or "3rd" in text_lower:
                    rank = 2
                
                # Always show top 3
                if len(results) >= 3:
                    return f"{time_label}: {results[0][0]} is first with {results[0][1]:,} words, {results[1][0]} is second with {results[1][1]:,} words, {results[2][0]} is third with {results[2][1]:,} words"
                else:
                    top_list = ", ".join([f"{r[0]} ({r[1]:,})" for r in results])
                    return f"Top talkers: {top_list}"
            return "No data"
        except Exception as e:
            print(f"Talkative error: {e}")
            return "Error"
    
    # Would you rather (AI-generated, diabolical)
    if "would you rather" in text_lower or "give me a would you rather" in text_lower or "give us a would you rather" in text_lower:
        try:
            import httpx
            prompt = "Generate ONE disturbing/dark 'would you rather' question with two terrible choices. Make it morally challenging and uncomfortable. Just the question, no explanation. Maximum 30 words."
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                question = response.json().get('response', '').strip()
                return f"[SPOKEN]{question}"
        except Exception as e:
            print(f"WYR error: {e}")
            return "[SPOKEN]Would you rather error occurred"
    
    # Bookmark conversation
    if "bookmark this" in text_lower or "save this conversation" in text_lower or "bookmark conversation" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Get last 5 minutes of conversation
            time_threshold = datetime.now() - timedelta(minutes=5)
            
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""SELECT transcription, username, timestamp FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        ORDER BY timestamp ASC""",
                     (str(guild_id), time_threshold.strftime('%Y-%m-%d %H:%M:%S')))
            results = c.fetchall()
            
            if results:
                bookmark_text = "\n".join([f"[{ts}] {user}: {text}" for text, user, ts in results])
                
                # Save bookmark
                c.execute("""CREATE TABLE IF NOT EXISTS bookmarks 
                            (guild_id TEXT, saved_by TEXT, bookmark_text TEXT, 
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
                c.execute("INSERT INTO bookmarks (guild_id, saved_by, bookmark_text) VALUES (?, ?, ?)",
                         (str(guild_id), requestor.name, bookmark_text))
                conn.commit()
                conn.close()
                
                return f"Bookmarked last {len(results)} messages"
            return "No recent conversation to bookmark"
        except Exception as e:
            print(f"Bookmark error: {e}")
            return "Error bookmarking"
    
    # Show bookmarks
    if "show bookmarks" in text_lower or "my bookmarks" in text_lower or "list bookmarks" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT timestamp, bookmark_text FROM bookmarks 
                        WHERE guild_id = ? AND saved_by = ?
                        ORDER BY timestamp DESC LIMIT 5""",
                     (str(guild_id), requestor.name))
            results = c.fetchall()
            conn.close()
            
            if results:
                bookmark_list = []
                for i, (ts, text) in enumerate(results, 1):
                    preview = text[:100].replace("\n", " ")
                    bookmark_list.append(f"{i}. [{ts}] {preview}...")
                
                return "Your bookmarks: " + " | ".join(bookmark_list)
            return "No bookmarks found"
        except Exception as e:
            print(f"Show bookmarks error: {e}")
            return "Error"
    
    # Search past conversations
    if "search past" in text_lower or "find conversation" in text_lower or "look back" in text_lower:
        try:
            # Extract search terms
            search_terms = text_lower.replace("search past", "").replace("find conversation", "").replace("look back", "").strip()
            
            if len(search_terms) < 3:
                return "What should I search for?"
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Search transcriptions
            c.execute("""SELECT username, transcription, timestamp 
                        FROM transcriptions 
                        WHERE guild_id = ? 
                        AND LOWER(transcription) LIKE ?
                        ORDER BY timestamp DESC 
                        LIMIT 5""",
                     (str(guild_id), f'%{search_terms}%'))
            results = c.fetchall()
            conn.close()
            
            if results:
                summaries = []
                for username, text, timestamp in results:
                    summaries.append(f"{username} ({timestamp}): {text[:50]}...")
                
                response = f"Found {len(results)} matches: " + " | ".join(summaries)
                return response
            return f"No matches for '{search_terms}'"
        except Exception as e:
            print(f"Search error: {e}")
            return "Search error"
    
    # Recent conversations summary
    if "what did we talk about" in text_lower or "conversation summary" in text_lower or "recent topics" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Determine time window
            if "today" in text_lower:
                time_window = timedelta(hours=24)
            elif "week" in text_lower:
                time_window = timedelta(days=7)
            else:
                time_window = timedelta(hours=2)
            
            time_threshold = datetime.now() - time_window
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND timestamp > ?
                        ORDER BY timestamp ASC""",
                     (str(guild_id), time_threshold.strftime('%Y-%m-%d %H:%M:%S')))
            results = c.fetchall()
            conn.close()
            
            if results:
                # Use AI to summarize
                all_text = " ".join([r[0] for r in results[:50]])  # Limit to avoid token overflow
                
                import httpx
                prompt = f"Summarize the key topics discussed in these conversations in 2-3 sentences: {all_text[:2000]}"
                
                async def _summarize():
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                        )
                        if response.status_code == 200:
                            return response.json().get('response', 'Unable to summarize')
                        return "Error"
                
                summary = await retry_async(_summarize, max_retries=2)
                return f"Recent topics: {summary}"
            return "No recent conversations"
        except Exception as e:
            print(f"Summary error: {e}")
            return "Error"
    
    # Smart queries - AI-powered database questions
    query_triggers = ["how many times", "how often", "how much", "who said", "what did", "when did", 
                     "who talked about", "most mentioned", "what are my", "top cuss", "top word",
                     "which word", "favorite cuss", "top swear", "all-time", "all time", "count", 
                     "how many", "number of times"]
    
    if any(trigger in text_lower for trigger in query_triggers):
        try:
            # Build context for AI
            user_context = ""
            if "my" in text_lower or "i " in text_lower:
                user_context = f"The person asking is username: {requestor.display_name}"
            
            # Ask AI to generate SQL
            import httpx
            # Add today filter if mentioned
            time_filter = ""
            if "today" in text_lower:
                from datetime import datetime
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                time_filter = f"AND timestamp >= '{today_start.strftime('%Y-%m-%d %H:%M:%S')}'"
            
            # Special handling for "how many times I said X"
            if "how many times" in text_lower and ("i said" in text_lower or "i've said" in text_lower):
                user_context = f"The person asking is user_id: {requestor.id}, username: {requestor.display_name}"
            
            prompt = f"""Given this SQLite database schema:
transcriptions table: user_id TEXT, username TEXT, display_name TEXT, guild_id TEXT, channel_id TEXT, transcription TEXT, timestamp DATETIME

Question: {text}
{user_context}

Generate a SELECT SQL query to answer this question. Return ONLY the SQL query, no explanation.
Use guild_id = '{guild_id}' in WHERE clause.
{f"Add this time filter: {time_filter}" if time_filter else ""}
For "top swear words" queries: Use CASE statements to extract and count individual profanity words. Add WHERE profanity_word IS NOT NULL (or equivalent) to exclude rows without profanity. Return top 3 non-null results.
For profanity counting, check for these words: fuck, fucking, fucked, fucker, fuckin, shit, shitting, shitty, damn, damned, bitch, bitching, bitches, ass, asshole, bastard, hell, crap, cock, dick, pussy, cunt, whore, slut, fag, faggot, retard, retarded, nigga, nigger, niggaz.
ONLY add timestamp filters if "today" is in the question AND time_filter parameter is set.
Format numbers with commas in the response."""

            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                )
            
            if response.status_code == 200:
                sql_query = response.json().get('response', '').strip()
                # Clean up SQL
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                
                print(f"📊 SQL: {sql_query}")
                
                conn = get_db_connection()
                c = conn.cursor()
                c.execute(sql_query)
                results = c.fetchall()
                conn.close()
                
                if results:
                    # Check if this is a "top swear words" query - format with counts
                    if "top swear" in text.lower() or "top cuss" in text.lower():
                        # Format as: word1 (count), word2 (count), word3 (count)
                        formatted_results = ", ".join([f"{row[0]} ({row[1]})" for row in results if row[0]])
                        return formatted_results

                    # Ask AI to format the response naturally
                    result_prompt = f"Question: {text}\nSQL Results: {results}\n\nAnswer in 10 words or less. Be direct and concise:"
                    
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": "gemma2:27b", "prompt": result_prompt, "stream": False}
                        )
                    
                    if response.status_code == 200:
                        answer = response.json().get('response', '').strip()
                        return answer
                
                return "No data found. Try rephrasing or ask a different question"
        except Exception as e:
            print(f"Smart query error: {e}")
            return None  # Fall through to AI
    
    # Save personal context
    if "remember that i" in text_lower or "remember i" in text_lower or "i like" in text_lower or "my favorite" in text_lower:
        try:
            # Extract the fact
            fact = None
            if "remember that i" in text_lower:
                fact = text_lower.split("remember that i", 1)[1].strip()
            elif "remember i" in text_lower:
                fact = text_lower.split("remember i", 1)[1].strip()
            elif "i like" in text_lower:
                fact = "likes " + text_lower.split("i like", 1)[1].strip()
            elif "my favorite" in text_lower:
                fact = "favorite is " + text_lower.split("my favorite", 1)[1].strip()
            
            if fact and len(fact) > 3:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""CREATE TABLE IF NOT EXISTS user_context 
                            (guild_id TEXT, user_id TEXT, username TEXT, context_key TEXT, 
                            context_value TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
                c.execute("""INSERT INTO user_context (guild_id, user_id, username, context_key, context_value) 
                            VALUES (?, ?, ?, ?, ?)""",
                         (str(guild_id), str(requestor.id), requestor.name, "personal", fact))
                conn.commit()
                conn.close()
                
                return f"I'll remember that you {fact}"
        except Exception as e:
            print(f"Context save error: {e}")
            return "Error saving"
    
    # Recall personal context
    if "what do you know about me" in text_lower or "what do i like" in text_lower or "my interests" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT context_value FROM user_context 
                        WHERE guild_id = ? AND user_id = ? 
                        ORDER BY timestamp DESC LIMIT 5""",
                     (str(guild_id), str(requestor.id)))
            results = c.fetchall()
            conn.close()
            
            if results:
                facts = [r[0] for r in results]
                return f"I know that you: {', '.join(facts)}"
            return "I don't have any saved info about you yet"
        except Exception as e:
            print(f"Context recall error: {e}")
            return "Error"
    
    # Comprehensive stats dashboard
    if "my stats" in text_lower or "my statistics" in text_lower or "show my stats" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Total messages
            c.execute("SELECT COUNT(*) FROM transcriptions WHERE guild_id = ? AND username = ?",
                     (str(guild_id), requestor.name))
            total_messages = c.fetchone()[0]
            
            # Total words
            c.execute("""SELECT SUM(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) 
                        FROM transcriptions WHERE guild_id = ? AND username = ?""",
                     (str(guild_id), requestor.name))
            total_words = c.fetchone()[0] or 0
            
            # Songs played
            c.execute("SELECT COUNT(*) FROM music_history WHERE guild_id = ? AND username = ?",
                     (str(guild_id), requestor.name))
            songs_played = c.fetchone()[0]
            
            # Profanity count
            profanity_words = ['fuck', 'shit', 'damn', 'bitch', 'ass', 'hell', 'crap']
            profanity_count = 0
            for word in profanity_words:
                c.execute(f"SELECT SUM(LENGTH(transcription) - LENGTH(REPLACE(LOWER(transcription), '{word}', ''))) / LENGTH('{word}') FROM transcriptions WHERE guild_id = ? AND username = ?",
                         (str(guild_id), requestor.name))
                result = c.fetchone()[0]
                if result:
                    profanity_count += int(result)
            
            conn.close()
            
            stats_text = f"{requestor.display_name}'s stats: {total_messages:,} messages, {total_words:,} words, {songs_played} songs played, {profanity_count} swears"
            
            # Speak it
            vc = voice_clients.get(guild_id, {}).get('vc')
            if vc:
                await asyncio.to_thread(speak_text, stats_text, vc, guild_id)
            
            return stats_text
        except Exception as e:
            print(f"Stats dashboard error: {e}")
            return "Error loading stats"
    
    # Leaderboards (week/month)
    if "leaderboard" in text_lower or "top talkers" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Determine time period
            if "week" in text_lower or "weekly" in text_lower:
                time_window = timedelta(days=7)
                period = "this week"
            elif "month" in text_lower or "monthly" in text_lower:
                time_window = timedelta(days=30)
                period = "this month"
            else:
                time_window = timedelta(days=7)
                period = "this week"
            
            time_threshold = datetime.now() - time_window
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get top talkers
            c.execute("""SELECT username, 
                        COUNT(*) as messages,
                        SUM(LENGTH(transcription) - LENGTH(REPLACE(transcription, ' ', '')) + 1) as words
                        FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        GROUP BY username 
                        ORDER BY words DESC 
                        LIMIT 5""",
                     (str(guild_id), time_threshold.strftime('%Y-%m-%d %H:%M:%S')))
            results = c.fetchall()
            conn.close()
            
            if results:
                leaderboard = f"Leaderboard {period}: "
                entries = []
                for i, (username, messages, words) in enumerate(results, 1):
                    entries.append(f"{i}. {username} ({words:,} words)")
                
                leaderboard += ", ".join(entries)
                return leaderboard
            return f"No data for {period}"
        except Exception as e:
            print(f"Leaderboard error: {e}")
            return "Error"
    
    # Reset my stats
    if "reset my stats" in text_lower or "reset my statistics" in text_lower or "delete my data" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Delete only this user's transcriptions
            c.execute("DELETE FROM transcriptions WHERE guild_id = ? AND username = ?",
                     (str(guild_id), requestor.name))
            
            deleted = c.rowcount
            conn.commit()
            conn.close()
            
            return f"Deleted {deleted:,} messages for {requestor.name}"
        except Exception as e:
            print(f"Reset error: {e}")
            return "Error resetting"
    
    # Lie detector (joke feature)
    if "lie detector" in text_lower or "detect lie" in text_lower or "is that true" in text_lower:
        import random
        
        # Get last message from someone else
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT username, transcription FROM transcriptions 
                        WHERE guild_id = ? AND username != ?
                        ORDER BY timestamp DESC LIMIT 1""",
                     (str(guild_id), requestor.name))
            result = c.fetchone()
            conn.close()
            
            if result:
                username, message = result
                
                # Random determination
                is_lie = random.choice([True, False])
                confidence = random.randint(60, 99)
                
                if is_lie:
                    response = f"LIAR! {username}'s statement registers as FALSE with {confidence}% confidence. I detect deception."
                else:
                    response = f"Truth detected. {username}'s statement appears genuine with {confidence}% confidence."
                
                # Speak it dramatically
                vc = voice_clients.get(guild_id, {}).get('vc')
                if vc:
                    await asyncio.to_thread(speak_text, response, vc, guild_id)
                
                return response
            return "No recent statements to analyze"
        except Exception as e:
            print(f"Lie detector error: {e}")
            return "Lie detector malfunction"
    
    # Sentiment analysis
    if "sentiment" in text_lower or "mood" in text_lower or "vibe" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get recent messages (last hour or last 50)
            if "today" in text_lower:
                from datetime import datetime, timedelta
                time_threshold = datetime.now() - timedelta(hours=24)
                c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ? AND timestamp > ?",
                         (str(guild_id), time_threshold.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ? ORDER BY timestamp DESC LIMIT 50",
                         (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                # Simple sentiment analysis
                all_text = " ".join([r[0].lower() for r in results])
                
                positive_words = ['good', 'great', 'awesome', 'love', 'happy', 'excellent', 'amazing', 'wonderful', 'fantastic', 'nice', 'lol', 'haha']
                negative_words = ['bad', 'terrible', 'hate', 'sad', 'awful', 'horrible', 'worst', 'angry', 
                             'fuck', 'shit', 'damn', 'bitch', 'ass', 'hell', 'crap', 'stupid', 'suck', 'sucks']
                
                pos_count = sum(all_text.count(word) for word in positive_words)
                neg_count = sum(all_text.count(word) for word in negative_words)
                
                if pos_count > neg_count * 1.5:
                    sentiment = "very positive"
                elif pos_count > neg_count:
                    sentiment = "positive"
                elif neg_count > pos_count * 1.5:
                    sentiment = "very negative"
                elif neg_count > pos_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                return f"Current vibe: {sentiment} ({pos_count} positive, {neg_count} negative words)"
            return "No recent data"
        except Exception as e:
            print(f"Sentiment error: {e}")
            return "Error analyzing"
    
    # Top cuss words / favorite cuss word
    if "top cuss" in text_lower or "favorite cuss" in text_lower or "top swear" in text_lower or "most used cuss" in text_lower:
        try:
            # Determine target
            target_user = requestor.name
            if "my" not in text_lower:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ? AND username = ?",
                     (str(guild_id), target_user))
            results = c.fetchall()
            conn.close()
            
            if results:
                from collections import Counter
                profanity_words = [
                    'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitting', 'shitty',
                    'damn', 'damned', 'bitch', 'bitching', 'ass', 'asshole', 'bastard',
                    'hell', 'crap', 'piss', 'pissed', 'cock', 'dick', 'pussy', 'cunt',
                    'whore', 'slut', 'fag', 'retard', 'nigger'
                ]
                
                all_words = []
                for row in results:
                    words = row[0].lower().split()
                    all_words.extend([w.strip('.,!?;:') for w in words if w.strip('.,!?;:') in profanity_words])
                
                if all_words:
                    counter = Counter(all_words)
                    top_5 = counter.most_common(5)
                    result_str = ", ".join([f"{word} ({number_to_words(count)})" for word, count in top_5])
                    return f"{target_user}'s top cuss words: {result_str}"
            return f"No profanity found for {target_user}"
        except Exception as e:
            print(f"Top cuss error: {e}")
            return "Error"
    
    # Stats - who swore most
    if any(pattern in text_lower for pattern in ["who swore", "who swear", "who cuss", "who curse", "potty mouth", "who says fuck"]):
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            profanity_words = [
                'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitting', 'shitty',
                'damn', 'damned', 'bitch', 'bitching', 'ass', 'asshole', 'bastard',
                'hell', 'crap', 'piss', 'pissed', 'cock', 'dick', 'pussy', 'cunt',
                'whore', 'slut', 'fag', 'retard', 'nigger'
            ]
            
            # Build CASE statement for counting - divide by word length to get occurrences
            case_parts = []
            for word in profanity_words:
                word_len = len(word)
                case_parts.append(f"(LENGTH(transcription) - LENGTH(REPLACE(LOWER(transcription), '{word}', ''))) / {word_len}")
            case_sql = " + ".join(case_parts)
            
            query = f"""SELECT username, SUM({case_sql}) as profanity_count 
                        FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY username 
                        ORDER BY profanity_count DESC 
                        LIMIT 3"""
            
            c.execute(query, (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if results and results[0][1] > 0:
                top3 = ", ".join([f"{r[0]} ({r[1]})" for r in results])
                response = f"Potty mouths: {top3}"
                
                return response
            return "No profanity found"
        except Exception as e:
            print(f"Stats error: {e}")
            return "Error"
    
    # Debug mode toggle (owner only)
    if "debug mode" in text_lower or "toggle debug" in text_lower:
        if requestor.id == 251202041966231572:
            global DEBUG_MODE
            DEBUG_MODE = not DEBUG_MODE
            status = "enabled" if DEBUG_MODE else "disabled"
            return f"Debug mode {status}"
        return "Owner only command"
    
    # Multi-server sync (owner only)
    if "enable sync" in text_lower or "sync servers" in text_lower:
        if requestor.id == 251202041966231572:
            SYNC_ENABLED = True
            if str(guild_id) not in SYNC_GUILDS:
                SYNC_GUILDS.append(str(guild_id))
            return f"Sync enabled. {len(SYNC_GUILDS)} servers synced"
        return "Owner only"
    
    if "disable sync" in text_lower:
        if requestor.id == 251202041966231572:
            SYNC_ENABLED = False
            return "Sync disabled"
        return "Owner only"
    
    # Buffer diagnostic (owner only)
    if "buffer status" in text_lower or "check buffers" in text_lower:
        if requestor.id == 251202041966231572:
            if guild_id in voice_clients and 'sink' in voice_clients[guild_id]:
                sink = voice_clients[guild_id]['sink']
                buffer_info = [(uid, len(data.get("buffer", []))) for uid, data in sink.user_buffers.items()]
                if buffer_info:
                    info_str = ", ".join([f"{uid}: {size}" for uid, size in buffer_info])
                    return f"Buffers: {info_str}"
                return "No buffers active"
            return "Not in voice"
        return "Owner only command"
    
    # Go to bed (owner only)
    if "go to bed" in text_lower or ("goodnight" in text_lower and "jarvis" in text_lower):
        if requestor.id == 251202041966231572:  # Your user ID
            vc = voice_clients.get(guild_id, {}).get('vc')
            if vc:
                await vc.disconnect()
                if guild_id in voice_clients:
                    del voice_clients[guild_id]
                return "Goodnight, going to sleep"
        return "Only the boss can put me to bed"
    
    # Join voice channel
    if "join" in text_lower and ("voice" in text_lower or "channel" in text_lower):
        # Extract channel name if provided
        channel_name = None
        if "join " in text_lower:
            channel_name = text_lower.split("join ", 1)[1].replace("voice", "").replace("channel", "").strip()
        
        # Find channel
        target_channel = None
        if channel_name:
            for channel in guild.voice_channels:
                if channel_name.lower() in channel.name.lower():
                    target_channel = channel
                    break
        
        if not target_channel and requestor.voice:
            target_channel = requestor.voice.channel
        
        if target_channel:
            try:
                if guild_id in voice_clients:
                    await voice_clients[guild_id]['vc'].disconnect()
                
                vc = await target_channel.connect(cls=voice_recv.VoiceRecvClient)
                sink = AudioSink(guild_id, target_channel.id)
                vc.listen(sink)
                
                voice_clients[guild_id] = {'vc': vc, 'sink': sink}
                conversation_history[guild_id] = []
                bot.loop.create_task(transcribe_loop(guild_id, sink))
                save_voice_state(guild_id, target_channel.id)
                
                return f"Joined {target_channel.name}"
            except Exception as e:
                print(f"Join error: {e}")
                return "Error joining"
        return "No channel specified"
    
    # Leave voice
    if "leave" in text_lower and ("voice" in text_lower or "channel" in text_lower):
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc:
            await vc.disconnect()
            if guild_id in voice_clients:
                del voice_clients[guild_id]
            return "Left voice"
        return "Not in voice"
    
    # Song completion stats
    if "completion" in text_lower and "stats" in text_lower or "skip rate" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT 
                        COUNT(*) as total,
                        SUM(completed) as completed,
                        COUNT(*) - SUM(completed) as skipped
                        FROM music_history 
                        WHERE guild_id = ?""",
                     (str(guild_id),))
            result = c.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                total, completed, skipped = result
                completion_rate = (completed / total * 100) if total > 0 else 0
                return f"Music stats: {completed}/{total} completed ({completion_rate:.1f}%), {skipped} skipped"
            return "No music history"
        except Exception as e:
            print(f"Completion stats error: {e}")
            return "Error"
    
    # Music preferences
    if "music stats" in text_lower or "music preferences" in text_lower or "top songs" in text_lower or "favorite songs" in text_lower:
        try:
            # Determine target user
            target_user = requestor.name
            if "my" not in text_lower:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Top requested songs
            c.execute("""SELECT song_title, COUNT(*) as plays 
                        FROM music_history 
                        WHERE guild_id = ? AND username = ?
                        GROUP BY song_title 
                        ORDER BY plays DESC 
                        LIMIT 5""",
                     (str(guild_id), target_user))
            results = c.fetchall()
            conn.close()
            
            if results:
                top_songs = ", ".join([f"{r[0][:30]} ({r[1]}x)" for r in results])
                return f"{target_user}'s top songs: {top_songs}"
            return f"No music history for {target_user}"
        except Exception as e:
            print(f"Music stats error: {e}")
            return "Error"
    
    # Set reminder
    if "remind me" in text_lower or "reminder" in text_lower:
        try:
            import re
            
            # Parse time
            time_match = re.search(r'(\d+)\s*(second|minute|hour|min|sec|hr)s?', text_lower)
            if time_match:
                amount = int(time_match.group(1))
                unit = time_match.group(2)
                
                # Convert to seconds
                if unit in ['second', 'sec']:
                    delay = amount
                elif unit in ['minute', 'min']:
                    delay = amount * 60
                elif unit in ['hour', 'hr']:
                    delay = amount * 3600
                else:
                    delay = amount * 60  # default to minutes
                
                # Extract reminder message
                message = text_lower.split("to ", 1)[1] if " to " in text_lower else "your reminder"
                
                # Create reminder task
                async def send_reminder():
                    await asyncio.sleep(delay)
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc:
                        reminder_text = f"{requestor.display_name}, reminder: {message}"
                        await asyncio.to_thread(speak_text, reminder_text, vc, guild_id)
                        print(f"⏰ Reminder: {reminder_text}")
                
                task = bot.loop.create_task(send_reminder())
                active_reminders.append(task)
                
                time_str = f"{amount} {unit}{'s' if amount > 1 else ''}"
                return f"Reminder set for {time_str}"
            
            return "Specify time like 'remind me in 10 minutes'"
        except Exception as e:
            print(f"Reminder error: {e}")
            return "Error setting reminder"
    
    # Quote of the day
    if "quote of the day" in text_lower or "random quote" in text_lower or "memorable quote" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get a random interesting quote (longer messages tend to be more interesting)
            c.execute("""SELECT username, transcription, timestamp 
                        FROM transcriptions 
                        WHERE guild_id = ? 
                        AND LENGTH(transcription) > 50 
                        AND LENGTH(transcription) < 200
                        ORDER BY RANDOM() 
                        LIMIT 1""",
                     (str(guild_id),))
            result = c.fetchone()
            conn.close()
            
            if result:
                username, quote, timestamp = result
                date = timestamp.split()[0]
                return f'Quote of the day from {username} on {date}: "{quote}"'
            return "No quotes found"
        except Exception as e:
            print(f"Quote error: {e}")
            return "Error"
    
    # Conversation streaks
    if "streak" in text_lower or "consecutive days" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Determine target
            target_user = requestor.name
            if "my" not in text_lower:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get distinct days with messages
            c.execute("""SELECT DISTINCT DATE(timestamp) as day 
                        FROM transcriptions 
                        WHERE guild_id = ? AND username = ?
                        ORDER BY day DESC""",
                     (str(guild_id), target_user))
            results = c.fetchall()
            conn.close()
            
            if results:
                days = [datetime.strptime(r[0], '%Y-%m-%d').date() for r in results]
                
                # Calculate current streak
                current_streak = 0
                today = datetime.now().date()
                
                for i, day in enumerate(days):
                    expected_day = today - timedelta(days=i)
                    if day == expected_day:
                        current_streak += 1
                    else:
                        break
                
                # Calculate longest streak
                longest_streak = 1
                temp_streak = 1
                for i in range(len(days) - 1):
                    if (days[i] - days[i+1]).days == 1:
                        temp_streak += 1
                        longest_streak = max(longest_streak, temp_streak)
                    else:
                        temp_streak = 1
                
                return f"{target_user}: {current_streak} day current streak, {longest_streak} day record"
            return f"No activity found for {target_user}"
        except Exception as e:
            print(f"Streak error: {e}")
            return "Error"
    
    # Word cloud data
    if "word cloud" in text_lower or "top 20 words" in text_lower or "most common words" in text_lower:
        try:
            # Determine target user
            target_user = requestor.name
            if "server" in text_lower or "everyone" in text_lower:
                target_user = None
            else:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            if target_user:
                c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ? AND username = ?",
                         (str(guild_id), target_user))
            else:
                c.execute("SELECT transcription FROM transcriptions WHERE guild_id = ?",
                         (str(guild_id),))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                from collections import Counter
                import re
                
                # Skip common words
                skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                             'of', 'with', 'by', 'from', 'up', 'about', 'into', 'is', 'are', 'was', 
                             'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                             'my', 'your', 'that', 'this', 'what', 'jarvis', 'travis'}
                
                all_words = []
                for row in results:
                    words = re.findall(r'\b[a-z]{3,}\b', row[0].lower())
                    all_words.extend([w for w in words if w not in skip_words])
                
                if all_words:
                    counter = Counter(all_words)
                    top_20 = counter.most_common(20)
                    words_str = ", ".join([f"{word}({count})" for word, count in top_20])
                    
                    label = target_user if target_user else "Server"
                    return f"{label} word cloud: {words_str}"
            return "No data"
        except Exception as e:
            print(f"Word cloud error: {e}")
            return "Error"
    
    # Mood tracking over time
    if "mood trend" in text_lower or "mood over time" in text_lower or "getting happier" in text_lower or "getting sadder" in text_lower:
        try:
            from datetime import datetime, timedelta
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get messages from last 7 days, grouped by day
            results_by_day = {}
            for days_ago in range(7):
                day_start = datetime.now() - timedelta(days=days_ago+1)
                day_end = datetime.now() - timedelta(days=days_ago)
                
                c.execute("""SELECT transcription FROM transcriptions 
                            WHERE guild_id = ? AND timestamp BETWEEN ? AND ?""",
                         (str(guild_id), day_start.strftime('%Y-%m-%d %H:%M:%S'), 
                          day_end.strftime('%Y-%m-%d %H:%M:%S')))
                results_by_day[days_ago] = c.fetchall()
            
            conn.close()
            
            # Analyze sentiment for each day
            positive_words = ['good', 'great', 'awesome', 'love', 'happy', 'excellent', 'amazing', 'wonderful', 'lol', 'haha']
            negative_words = ['bad', 'terrible', 'hate', 'sad', 'awful', 'horrible', 'worst', 'angry', 'fuck', 'shit']
            
            daily_scores = []
            for days_ago in range(7):
                if results_by_day[days_ago]:
                    text = " ".join([r[0].lower() for r in results_by_day[days_ago]])
                    pos = sum(text.count(word) for word in positive_words)
                    neg = sum(text.count(word) for word in negative_words)
                    score = pos - neg
                    daily_scores.append(score)
            
            if len(daily_scores) >= 2:
                # Compare recent vs older
                recent = sum(daily_scores[:3]) / 3
                older = sum(daily_scores[3:]) / max(len(daily_scores[3:]), 1)
                
                if recent > older + 2:
                    trend = "getting happier"
                elif recent < older - 2:
                    trend = "getting sadder"
                else:
                    trend = "staying about the same"
                
                return f"Mood trend: {trend} (recent score: {recent:.1f}, previous: {older:.1f})"
            return "Not enough data for trend"
        except Exception as e:
            print(f"Mood tracking error: {e}")
            return "Error"
    
    # Extract conversation topics
    if "what topics" in text_lower or "conversation topics" in text_lower or "what did we discuss" in text_lower or "discussion topics" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Determine time window
            if "today" in text_lower:
                time_window = timedelta(hours=24)
            elif "week" in text_lower:
                time_window = timedelta(days=7)
            else:
                time_window = timedelta(hours=6)
            
            time_threshold = datetime.now() - time_window
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT transcription FROM transcriptions 
                        WHERE guild_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT 100""",
                     (str(guild_id), time_threshold.strftime('%Y-%m-%d %H:%M:%S')))
            results = c.fetchall()
            conn.close()
            
            if results:
                # Use AI to extract topics
                all_text = " ".join([r[0] for r in results[:50]])[:3000]
                
                import httpx
                prompt = f"Extract 5 main conversation topics from these messages as a short comma-separated list: {all_text}"
                
                async def _extract_topics():
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": "gemma2:27b", "prompt": prompt, "stream": False}
                        )
                        if response.status_code == 200:
                            return response.json().get('response', 'Unable to extract')
                        return "Error"
                
                topics = await retry_async(_extract_topics, max_retries=2)
                return f"Discussion topics: {topics}"
            return "No recent conversations"
        except Exception as e:
            print(f"Topics error: {e}")
            return "Error"
    
    # Voice activity heatmap
    if "when" in text_lower and ("most active" in text_lower or "people talk" in text_lower) or "activity heatmap" in text_lower:
        try:
            from datetime import datetime
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get messages by hour
            c.execute("""SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                        FROM transcriptions 
                        WHERE guild_id = ?
                        GROUP BY hour
                        ORDER BY hour""",
                     (str(guild_id),))
            results = c.fetchall()
            conn.close()
            
            if results:
                # Create hour map
                hour_data = {int(r[0]): r[1] for r in results}
                
                # Find peak hours
                sorted_hours = sorted(hour_data.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_hours[:3]
                
                def format_hour(h):
                    if h == 0:
                        return "12am"
                    elif h < 12:
                        return f"{h}am"
                    elif h == 12:
                        return "12pm"
                    else:
                        return f"{h-12}pm"
                
                peak_times = ", ".join([f"{format_hour(h)} ({count} msgs)" for h, count in top_3])
                return f"Most active times: {peak_times}"
            return "No activity data"
        except Exception as e:
            print(f"Heatmap error: {e}")
            return "Error"
    
    # AI image generation
    if any(trigger in text_lower for trigger in ["draw", "generate image", "create image", "picture of", "draw me"]):
        try:
            # Extract the prompt
            prompt = None
            if "draw me " in text_lower:
                prompt = text_lower.split("draw me ", 1)[1]
            elif "draw " in text_lower:
                prompt = text_lower.split("draw ", 1)[1]
            elif "picture of " in text_lower:
                prompt = text_lower.split("picture of ", 1)[1]
            elif "generate image " in text_lower:
                prompt = text_lower.split("generate image ", 1)[1]
            elif "create image " in text_lower:
                prompt = text_lower.split("create image ", 1)[1]
            
            if prompt and len(prompt) > 3:
                # Use Ollama to improve the prompt
                import httpx
                improve_prompt = f"Rewrite this as a detailed Stable Diffusion prompt (50 words max): {prompt}"
                
                async def _get_prompt():
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        response = await client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={"model": "gemma2:27b", "prompt": improve_prompt, "stream": False}
                        )
                        if response.status_code == 200:
                            return response.json().get('response', prompt)
                        return prompt
                
                enhanced_prompt = await retry_async(_get_prompt, max_retries=2)
                
                return f"Image generation request received: '{enhanced_prompt}'. (Feature requires Stable Diffusion API setup)"
            
            return "What should I draw?"
        except Exception as e:
            print(f"Image gen error: {e}")
            return "Error"
    
    # Uptime/version info
    if "uptime" in text_lower or "how long" in text_lower and "running" in text_lower or "version" in text_lower:
        from datetime import datetime
        if bot_start_time:
            uptime = datetime.now() - bot_start_time
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            
            uptime_str = f"Version {BOT_VERSION}. Uptime: "
            if days > 0:
                uptime_str += f"{days}d "
            uptime_str += f"{hours}h {minutes}m"
            
            return uptime_str
        return f"Version {BOT_VERSION}"
    
    # Help command
    if text_lower in ['help', 'help me', 'what can you do', 'commands', 'show commands']:
        help_text = """I can:
- Music: play, pause, skip, volume up/down, random song, similar song, repeat, shuffle queue
- Voice: mute/unmute users, change voice accent
- Stats: my stats, who swore most, who talked most, top words, leaderboard
- Smart queries: "how many times did I say X", sentiment analysis
- Memory: bookmark conversations, search past chats, remember personal info
- Translation: say X in Spanish
- Fun: would you rather, lie detector
- Admin: go to bed (owner), debug mode (owner)
Say 'Jarvis' before commands!"""
        
        # Speak it
        vc = voice_clients.get(guild_id, {}).get('vc')
        if vc:
            await asyncio.to_thread(speak_text, "I can help with music, stats, translation, and much more. Check the chat for full list.", vc, guild_id)
        
        return help_text
    
    # Undo last action
    if "undo" in text_lower or "undo that" in text_lower or "go back" in text_lower:
        if guild_id in last_actions:
            action = last_actions[guild_id]
            
            if action['action'] == 'blacklist':
                try:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("DELETE FROM blacklisted_songs WHERE guild_id = ? AND song_title = ? ORDER BY timestamp DESC LIMIT 1",
                             (str(guild_id), action['data']['song']))
                    conn.commit()
                    conn.close()
                    del last_actions[guild_id]
                    return f"Removed from blacklist: {action['data']['song'][:50]}"
                except:
                    return "Error undoing"
            
            elif action['action'] == 'skip':
                return "Cannot undo skip"
            
            elif action['action'] == 'volume':
                vc = voice_clients.get(guild_id, {}).get('vc')
                if vc and vc.source:
                    vc.source.volume = action['data']['old_volume']
                    del last_actions[guild_id]
                    return f"Volume restored to {int(action['data']['old_volume'] * 100)}%"
                return "No music playing"
            
            return "Cannot undo that action"
        return "Nothing to undo"
    
    # Set transcription language
    if "transcribe in" in text_lower or "language" in text_lower and "set" in text_lower:
        languages = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja',
            'chinese': 'zh', 'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi'
        }
        
        for lang_name, lang_code in languages.items():
            if lang_name in text_lower:
                transcription_language[guild_id] = lang_code
                return f"Transcription language set to {lang_name}"
        
        return "Supported: English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Chinese, Korean, Arabic, Hindi"
    
    # Wake word sensitivity
    if "wake word" in text_lower and ("strict" in text_lower or "loose" in text_lower or "sensitivity" in text_lower):
        if requestor.id == 251202041966231572:  # Owner only
            global wake_word_strict
            if "strict" in text_lower:
                wake_word_strict = True
                return "Wake word set to strict mode (exact match required)"
            elif "loose" in text_lower:
                wake_word_strict = False
                return "Wake word set to loose mode (partial match allowed)"
        return "Owner only command"
    
    # Voice effects
    if "voice effect" in text_lower or "add effect" in text_lower:
        if guild_id not in voice_settings:
            voice_settings[guild_id] = {}
        if 'effects' not in voice_settings[guild_id]:
            voice_settings[guild_id]['effects'] = {}
        
        if "echo" in text_lower:
            voice_settings[guild_id]['effects']['echo'] = True
            return "Echo effect enabled"
        elif "reverb" in text_lower:
            voice_settings[guild_id]['effects']['reverb'] = True
            return "Reverb effect enabled"
        elif "higher" in text_lower or "pitch up" in text_lower:
            voice_settings[guild_id]['effects']['pitch'] = 2
            return "Pitch increased"
        elif "lower" in text_lower or "pitch down" in text_lower:
            voice_settings[guild_id]['effects']['pitch'] = -2
            return "Pitch decreased"
        return "Say: echo, reverb, higher, or lower"
    
    if "clear effects" in text_lower or "remove effects" in text_lower or "no effects" in text_lower:
        if guild_id in voice_settings and 'effects' in voice_settings[guild_id]:
            voice_settings[guild_id]['effects'] = {}
            return "Voice effects cleared"
        return "No effects active"
    
    # Voice personality/accent switching
    if "change voice" in text_lower or "change accent" in text_lower or "switch voice" in text_lower or "use" in text_lower and "accent" in text_lower:
        try:
            accents = {
                'british': 'co.uk',
                'american': 'com',
                'australian': 'com.au',
                'indian': 'co.in',
                'irish': 'ie',
                'south african': 'co.za',
                'canadian': 'ca'
            }
            
            selected_accent = 'com'  # default
            for accent_name, accent_code in accents.items():
                if accent_name in text_lower:
                    selected_accent = accent_code
                    break
            
            if guild_id not in voice_settings:
                voice_settings[guild_id] = {}
            
            if 'personality' not in voice_settings[guild_id]:
                voice_settings[guild_id]['personality'] = {}
            
            voice_settings[guild_id]['personality']['accent'] = selected_accent
            
            accent_name = [k for k, v in accents.items() if v == selected_accent][0] if selected_accent != 'com' else 'american'
            return f"Voice changed to {accent_name} accent"
        except Exception as e:
            print(f"Voice change error: {e}")
            return "Error changing voice"
    
    # Play by genre - checks history first, falls back to YouTube
    genre_keywords = {
        'rap': ['rap', 'hip hop', 'rapper'],
        'hip hop': ['hip hop', 'rap', 'rapper'],
        'rock': ['rock', 'band'],
        'metal': ['metal', 'metalcore', 'heavy'],
        'pop': ['pop', 'chart', 'hits'],
        'country': ['country', 'nashville'],
        'jazz': ['jazz', 'smooth', 'blues'],
        'edm': ['edm', 'electronic', 'dubstep', 'house'],
        'reggae': ['reggae', 'ska'],
        'rnb': ['r&b', 'rnb', 'soul'],
        'indie': ['indie', 'alternative']
    }
    
    for genre, keywords in genre_keywords.items():
        if genre in text_lower and any(trigger in text_lower for trigger in ['play', 'put on', 'music', 'song']):
            try:
                conn = get_db_connection()
                c = conn.cursor()
                
                # Build query for songs matching genre keywords
                like_conditions = ' OR '.join(['LOWER(song_title) LIKE ?' for _ in keywords])
                like_params = [f'%{kw}%' for kw in keywords]
                
                # Get current song to exclude it
                current = currently_playing.get(guild_id, {}).get('title', '')
                
                c.execute(f"""
                    SELECT DISTINCT song_title 
                    FROM music_history 
                    WHERE guild_id = ? 
                    AND ({like_conditions})
                    AND song_title != ?
                    AND song_title NOT IN (
                        SELECT song_title 
                        FROM music_history 
                        WHERE guild_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 15
                    )
                    ORDER BY RANDOM() 
                    LIMIT 1
                """, [str(guild_id)] + like_params + [current, str(guild_id)])
                
                result = c.fetchone()
                
                # Also count how many total songs we have for this genre
                c.execute(f"""
                    SELECT COUNT(DISTINCT song_title) 
                    FROM music_history 
                    WHERE guild_id = ? 
                    AND ({like_conditions})
                """, [str(guild_id)] + like_params)
                total_count = c.fetchone()[0]
                conn.close()
                
                # If we have fewer than 5 songs for this genre, search YouTube instead
                if total_count < 5:
                    result = None
                    print(f"⚠️ Only {total_count} {genre} songs in history, searching YouTube for variety")
                
                if result and result[0] != current:
                    # Found in history!
                    query = result[0]
                    print(f"🎵 {genre.upper()} from history: {query}")
                    song_result = await search_youtube(query, guild_id)
                else:
                    # Not in history, search YouTube
                    import random
                    searches = [f'{genre} music', f'popular {genre}', f'{genre} hits']
                    query = random.choice(searches)
                    print(f"🔍 {genre.upper()} from YouTube: {query}")
                    song_result = await search_youtube(query, guild_id)
                
                if song_result:
                    if guild_id not in music_queues:
                        music_queues[guild_id] = []
                    music_queues[guild_id].append(song_result)
                    
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc and not vc.is_playing():
                        await play_next(guild_id)
                    
                    source = "from history" if result else "new"
                    return f"{genre.upper()} ({source}): {song_result['title'][:40]}"
            except Exception as e:
                print(f"Genre error: {e}")
                break
    
    # Random song from history
    if "random song" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""
                SELECT song_title FROM music_history 
                WHERE guild_id = ? 
                AND song_title NOT IN (
                    SELECT song_title FROM blacklisted_songs WHERE guild_id = ?
                )
                ORDER BY RANDOM() 
                LIMIT 1
            """, (str(guild_id), str(guild_id)))
            result = c.fetchone()
            conn.close()
            
            if result:
                query = result[0]
                print(f"🔀 Random: {query}")
                song_result = await search_youtube(query, guild_id)
                
                if song_result:
                    if guild_id not in music_queues:
                        music_queues[guild_id] = []
                    music_queues[guild_id].append(song_result)
                    
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc and not vc.is_playing():
                        await play_next(guild_id)
                    
                    return f"Random: {song_result['title']}"
            return "No history"
        except Exception as e:
            print(f"Random error: {e}")
            return "Error"
    
    # Similar song
    if "similar song" in text_lower or "play something similar" in text_lower or "play similar" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT song_title FROM music_history WHERE guild_id = ? ORDER BY timestamp DESC LIMIT 1", (str(guild_id),))
            result = c.fetchone()
            conn.close()
            
            if result:
                last_song = result[0]
                query = f"{last_song} similar"
                print(f"🔍 Similar to: {last_song}")
                song_result = await search_youtube(query, guild_id)
                
                if song_result:
                    if guild_id not in music_queues:
                        music_queues[guild_id] = []
                    music_queues[guild_id].append(song_result)
                    
                    # Save to history
                    try:
                        conn = get_db_connection()
                        c = conn.cursor()
                        c.execute("INSERT INTO music_history (guild_id, username, song_title) VALUES (?, ?, ?)", (str(guild_id), requestor.name, song_result['title']))
                        conn.commit()
                        conn.close()
                    except:
                        pass
                    
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc and not vc.is_playing():
                        await play_next(guild_id)
                    
                    return f"Similar: {song_result['title']}"
            return "No history"
        except Exception as e:
            print(f"Similar error: {e}")
            return "Error"
    
    # Play last song
    if "play the last song" in text_lower or "play last song" in text_lower or "replay that" in text_lower or "play that again" in text_lower or "repeat that song" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT song_title FROM music_history WHERE guild_id = ? ORDER BY timestamp DESC LIMIT 1",
                     (str(guild_id),))
            result = c.fetchone()
            conn.close()
            
            if result:
                query = result[0]
                print(f"🔁 Replaying: {query}")
                song_result = await search_youtube(query, guild_id)
                
                if song_result:
                    if guild_id not in music_queues:
                        music_queues[guild_id] = []
                    music_queues[guild_id].append(song_result)
                    
                    # Save to history
                    try:
                        conn = get_db_connection()
                        c = conn.cursor()
                        c.execute("INSERT INTO music_history (guild_id, username, song_title) VALUES (?, ?, ?)",
                                 (str(guild_id), requestor.name, song_result['title']))
                        conn.commit()
                        conn.close()
                    except:
                        pass
                    
                    vc = voice_clients.get(guild_id, {}).get('vc')
                    if vc and not vc.is_playing():
                        await play_next(guild_id)
                    
                    return f"Replaying: {song_result['title']}"
            return "No previous song"
        except Exception as e:
            print(f"Replay error: {e}")
            return "Error"
    
    # Play from Spotify
    if sp and ("play my" in text_lower or "play spotify" in text_lower):
        try:
            if "liked" in text_lower or "favorites" in text_lower or "saved" in text_lower:
                # Get random liked song
                results = sp.current_user_saved_tracks(limit=50)
                if results['items']:
                    import random
                    track = random.choice(results['items'])['track']
                    query = f"{track['name']} {track['artists'][0]['name']}"
                    print(f"🎵 Spotify: {query}")
                    
                    song_result = await search_youtube(query, guild_id)
                    if song_result:
                        if guild_id not in music_queues:
                            music_queues[guild_id] = []
                        music_queues[guild_id].append(song_result)
                        
                        vc = voice_clients.get(guild_id, {}).get('vc')
                        if vc and not vc.is_playing():
                            await play_next(guild_id)
                        
                        return f"Spotify pick: {song_result['title']}"
            return "Spotify feature needs setup"
        except Exception as e:
            print(f"Spotify error: {e}")
            return "Spotify error"
    
    # Play music (with aliases)
    # Check for play commands but exclude "never play this"
    play_triggers = ['play ', 'put on', 'queue', 'add', 'throw on']
    if any(trigger in text_lower for trigger in play_triggers) and "never play" not in text_lower:
        if is_music_blocked(requestor.name):
            return "You're not allowed to use music commands"
        
        # Remove "play" and clean up punctuation
        query = text_lower.replace("play", "", 1).strip()
        query = query.lstrip(',').strip()  # Remove leading comma
        if not query:
            return None
        print(f"🔍 Searching: {query}")
        song_result = await search_youtube(query, guild_id)
        
        if song_result:
            if guild_id not in music_queues:
                music_queues[guild_id] = []
            music_queues[guild_id].append(song_result)
            
            # Save to history
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS music_history (guild_id TEXT, username TEXT, song_title TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, completed INTEGER DEFAULT 0)")
                c.execute("INSERT INTO music_history (guild_id, username, song_title) VALUES (?, ?, ?)", (str(guild_id), requestor.name, song_result['title']))
                conn.commit()
                conn.close()
            except:
                pass
            
            vc = voice_clients.get(guild_id, {}).get('vc')
            if vc and not vc.is_playing():
                await play_next(guild_id)
            
            return f"Playing: {song_result['title']}"
        return "Not found"
    
    # Identity check - who am I
    if text_lower in ["who am i", "who am i?", "what's my name", "whats my name"]:
        return f"You are {requestor.display_name} (username: {requestor.name})"
    
    # Nickname system - "call me X"
    if "call me" in text_lower:
        try:
            # Extract nickname after "call me"
            nickname = text.lower().split("call me", 1)[1].strip()
            
            if not nickname or len(nickname) < 2:
                return "What should I call you?"
            
            # Capitalize first letter
            nickname = nickname[0].upper() + nickname[1:]
            
            # Store in database
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS nicknames 
                        (user_id TEXT PRIMARY KEY, username TEXT, nickname TEXT, 
                         guild_id TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            
            c.execute("""INSERT OR REPLACE INTO nicknames (user_id, username, nickname, guild_id) 
                        VALUES (?, ?, ?, ?)""",
                     (str(requestor.id), requestor.name, nickname, str(guild_id)))
            conn.commit()
            conn.close()
            
            return f"Got it, I'll call you {nickname}"
            
        except Exception as e:
            print(f"Nickname error: {e}")
            return "Error setting nickname"
    
    # Check nickname - "who am I" or "what's my name"
    if text_lower in ["who am i", "who am i?", "what's my name", "whats my name", "my name"]:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT nickname FROM nicknames WHERE user_id = ? AND guild_id = ?",
                     (str(requestor.id), str(guild_id)))
            result = c.fetchone()
            conn.close()
            
            if result:
                return f"I call you {result[0]} (real name: {requestor.display_name})"
            else:
                return f"You're {requestor.display_name}, no nickname set yet"
                
        except Exception as e:
            print(f"Name check error: {e}")
            return f"You're {requestor.display_name}"
    
    # Energy drop detector with proactive topic suggestion
    if "energy dropping" in text_lower or "suggest topic" in text_lower or "conversation help" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get recent activity
            c.execute("""
                SELECT timestamp FROM transcriptions
                WHERE guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (str(guild_id),))
            
            recent = c.fetchall()
            
            if len(recent) < 10:
                return "Not enough recent activity"
            
            from datetime import datetime
            
            # Calculate message rate (messages per minute)
            first = datetime.strptime(recent[-1][0], '%Y-%m-%d %H:%M:%S')
            last = datetime.strptime(recent[0][0], '%Y-%m-%d %H:%M:%S')
            time_diff = (last - first).total_seconds() / 60
            current_rate = len(recent) / time_diff if time_diff > 0 else 0
            
            # Get topics from last hour
            c.execute("""
                SELECT transcription FROM transcriptions
                WHERE guild_id = ?
                AND datetime(timestamp) > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 100
            """, (str(guild_id),))
            
            past_topics = c.fetchall()
            conn.close()
            
            if current_rate < 0.5:  # Less than 1 msg per 2 minutes = energy dropping
                # Extract common words/topics from recent history
                all_text = " ".join([t[0].lower() for t in past_topics])
                words = all_text.split()
                
                # Filter to interesting words (nouns, verbs)
                interesting = [w for w in words if len(w) > 4 and w not in 
                              ['jarvis', 'travis', 'would', 'could', 'should', 'about', 'think', 'really']]
                
                from collections import Counter
                common = Counter(interesting).most_common(5)
                
                if common:
                    topics = [word for word, count in common if count > 2]
                    
                    if topics:
                        import random
                        suggested_topic = random.choice(topics)
                        return f"⚠️ Energy dropping (only {current_rate:.1f} msg/min). Want to revisit '{suggested_topic}'?"
                    else:
                        return f"⚠️ Energy dropping ({current_rate:.1f} msg/min). Need a new topic?"
                else:
                    return "Energy low, but no clear topics to suggest"
            else:
                return f"Energy is fine ({current_rate:.1f} msg/min)"
            
        except Exception as e:
            print(f"Energy drop error: {e}")
            return "Error detecting energy drop"
    
    # Tilt Detector - gaming rage detection
    if any(phrase in text_lower for phrase in ['tilt', 'tilted', 'tilt check', 'rage check', 'is tilted', 'getting tilted', 'tilting', 'am i tilt', 'my tilt']):
        try:
            # Determine target user - check for "is X tilted" pattern first
            target_user = None
            
            # Pattern: "is [name] tilted"
            import re
            match = re.search(r'is (\w+) tilt', text_lower)
            if match:
                name_mentioned = match.group(1)
                for member in guild.members:
                    if member.name.lower() == name_mentioned or member.display_name.lower() == name_mentioned:
                        target_user = member.name
                        break
            
            # Fallback: check if any member name is in the text
            if not target_user:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            # Default to requestor
            if not target_user:
                target_user = requestor.name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get last 20 messages from target
            c.execute("""
                SELECT transcription, timestamp FROM transcriptions
                WHERE guild_id = ? AND username = ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (str(guild_id), target_user))
            
            recent = c.fetchall()
            conn.close()
            
            if len(recent) < 5:
                return f"Not enough data on {target_user}"
            
            from datetime import datetime
            
            # Tilt indicators
            tilt_score = 0
            
            # 1. Profanity frequency (rage indicator)
            profanity_words = ['fuck', 'shit', 'damn', 'bitch', 'ass', 'hell', 'god', 'christ']
            profanity_count = sum(1 for msg in recent for word in profanity_words if word in msg[0].lower())
            tilt_score += profanity_count * 5
            
            # 2. Message frequency (rapid typing = tilted)
            if len(recent) >= 10:
                first = datetime.strptime(recent[9][1], '%Y-%m-%d %H:%M:%S')
                last = datetime.strptime(recent[0][1], '%Y-%m-%d %H:%M:%S')
                time_diff = (last - first).total_seconds() / 60
                msg_rate = 10 / time_diff if time_diff > 0 else 0
                
                if msg_rate > 3:  # More than 3 msgs/min = rapid fire
                    tilt_score += 20
            
            # 3. Short aggressive messages (spam = tilt)
            short_msgs = sum(1 for msg in recent if len(msg[0]) < 15)
            if short_msgs > len(recent) * 0.6:  # 60%+ short = tilted
                tilt_score += 15
            
            # 4. Gaming keywords (specific frustration)
            tilt_phrases = ['bullshit', 'rigged', 'lucky', 'broken', 'op', 'nerf', 'trash', 'garbage', 'noob']
            game_rage = sum(1 for msg in recent for phrase in tilt_phrases if phrase in msg[0].lower())
            tilt_score += game_rage * 8
            
            # Cap at 100
            tilt_score = min(100, tilt_score)
            
            # Generate assessment
            if tilt_score >= 75:
                assessment = "🔥 FULL TILT - Take a break NOW"
                recommendation = "5 min break recommended"
            elif tilt_score >= 50:
                assessment = "😤 Getting Tilted - Watch yourself"
                recommendation = "Deep breath, focus"
            elif tilt_score >= 25:
                assessment = "😐 Slightly Frustrated - Under control"
                recommendation = "You're fine, keep playing"
            else:
                assessment = "😌 Chill Vibes - No tilt detected"
                recommendation = "Mental is strong"
            
            return f"{target_user}'s Tilt: {tilt_score}% - {assessment}. {recommendation}"
            
        except Exception as e:
            print(f"Tilt error: {e}")
            return "Error checking tilt"
    
    # Conversation DNA Analyzer
    if "dna" in text_lower or "personality profile" in text_lower or "behavioral profile" in text_lower:
        try:
            # Determine target user - check for "is X tilted" pattern first
            target_user = None
            
            # Pattern: "is [name] tilted"
            import re
            match = re.search(r'is (\w+) tilt', text_lower)
            if match:
                name_mentioned = match.group(1)
                for member in guild.members:
                    if member.name.lower() == name_mentioned or member.display_name.lower() == name_mentioned:
                        target_user = member.name
                        break
            
            # Fallback: check if any member name is in the text
            if not target_user:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            # Default to requestor
            if not target_user:
                target_user = requestor.name
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Gather comprehensive behavioral data
            c.execute("""
                SELECT transcription, timestamp, 
                       strftime('%H', timestamp) as hour,
                       LENGTH(transcription) as msg_length
                FROM transcriptions
                WHERE guild_id = ? AND username = ?
                ORDER BY timestamp DESC
                LIMIT 500
            """, (str(guild_id), target_user))
            
            data = c.fetchall()
            conn.close()
            
            if len(data) < 50:
                return f"Need more data on {target_user} (only {len(data)} messages)"
            
            from collections import Counter
            import re
            
            # Build DNA profile
            all_text = " ".join([d[0].lower() for d in data])
            words = all_text.split()
            
            # 1. Humor style (dark, sarcastic, wholesome)
            dark_humor = sum(1 for w in words if w in ['dead', 'kill', 'murder', 'death', 'corpse'])
            sarcasm = sum(1 for w in words if w in ['sure', 'totally', 'obviously', 'definitely'])
            wholesome = sum(1 for w in words if w in ['love', 'awesome', 'nice', 'great', 'happy'])
            
            humor_total = dark_humor + sarcasm + wholesome + 1
            humor_style = max([
                (dark_humor/humor_total*100, "Dark"),
                (sarcasm/humor_total*100, "Sarcastic"),
                (wholesome/humor_total*100, "Wholesome")
            ])
            
            # 2. Aggression level
            profanity = sum(1 for w in words if w in ['fuck', 'shit', 'damn', 'bitch', 'ass'])
            aggression = min(100, (profanity / len(data)) * 100)
            
            # 3. Vocabulary complexity
            unique_words = len(set(words))
            avg_word_len = sum(len(w) for w in words) / len(words)
            complexity = min(100, (unique_words / len(words)) * 100 + avg_word_len * 10)
            
            # 4. Activity pattern (night owl vs morning person)
            hours = [int(d[2]) for d in data if d[2]]
            avg_hour = sum(hours) / len(hours) if hours else 12
            time_type = "Night Owl 🦉" if avg_hour >= 20 or avg_hour <= 4 else "Morning Person ☀️" if avg_hour <= 11 else "Afternoon Vibes 🌤️"
            
            # 5. Conversation starter vs responder
            # Simple heuristic: longer messages = starter
            avg_length = sum(d[3] for d in data) / len(data)
            role = "Conversation Starter 🎬" if avg_length > 50 else "Quick Responder ⚡"
            
            # 6. Emotional volatility
            msg_lengths = [d[3] for d in data]
            volatility = (max(msg_lengths) - min(msg_lengths)) / (sum(msg_lengths)/len(msg_lengths))
            volatility_score = min(100, volatility * 10)
            
            # Generate DNA summary
            result = f"🧬 {target_user}'s Conversation DNA:\n"
            result += f"  Humor: {humor_style[1]} ({humor_style[0]:.0f}%)\n"
            result += f"  Aggression: {aggression:.0f}%\n"
            result += f"  Vocabulary: {complexity:.0f}% complexity\n"
            result += f"  Active Time: {time_type} (avg {avg_hour:.0f}:00)\n"
            result += f"  Role: {role}\n"
            result += f"  Volatility: {volatility_score:.0f}% (consistency)\n"
            result += f"\nPrediction: Next message likely in {avg_hour:.0f}:00 hour range, {avg_length:.0f} chars"
            
            return result
            
        except Exception as e:
            print(f"DNA error: {e}")
            return "Error analyzing DNA"
    
    # Response time tracker - who replies fastest
    if "response time" in text_lower or "who replies fastest" in text_lower or "quickest responder" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""
                SELECT username, timestamp FROM transcriptions
                WHERE guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 500
            """, (str(guild_id),))
            
            messages = c.fetchall()
            conn.close()
            
            from collections import defaultdict
            from datetime import datetime
            
            response_times = defaultdict(list)
            prev_user = None
            prev_time = None
            
            for username, timestamp in reversed(messages):
                if prev_user and prev_user != username and prev_time:
                    current = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    previous = datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S')
                    diff = (current - previous).total_seconds()
                    
                    if diff < 30:
                        response_times[username].append(diff)
                
                prev_user = username
                prev_time = timestamp
            
            avg_times = {user: sum(times)/len(times) for user, times in response_times.items() if len(times) > 5}
            
            if not avg_times:
                return "Need more response data"
            
            fastest = sorted(avg_times.items(), key=lambda x: x[1])[:5]
            
            result = "⚡ Fastest responders:\n"
            for user, avg_time in fastest:
                result += f"  {user}: {avg_time:.1f}s avg\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Response time error: {e}")
            return "Error tracking response times"
    
    # Topic diversity - who brings up unique topics
    if "topic diversity" in text_lower or "unique topics" in text_lower or "conversation diversity" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""
                SELECT username, transcription FROM transcriptions
                WHERE guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (str(guild_id),))
            
            messages = c.fetchall()
            conn.close()
            
            from collections import defaultdict
            
            user_words = defaultdict(set)
            
            for username, transcription in messages:
                words = transcription.lower().split()
                # Extract meaningful words (nouns, longer words)
                meaningful = [w.strip('.,!?;:') for w in words if len(w) > 4]
                user_words[username].update(meaningful)
            
            if len(user_words) < 2:
                return "Need more users"
            
            diversity_scores = {user: len(words) for user, words in user_words.items()}
            top_diverse = sorted(diversity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = "🎨 Most diverse vocabularies:\n"
            for user, unique_count in top_diverse:
                result += f"  {user}: {unique_count} unique words\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Topic diversity error: {e}")
            return "Error analyzing diversity"
    
    # Influence score - who drives conversation
    if "influence" in text_lower or "who drives" in text_lower or "conversation leader" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""
                SELECT username, transcription, timestamp FROM transcriptions
                WHERE guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 500
            """, (str(guild_id),))
            
            messages = c.fetchall()
            conn.close()
            
            from collections import defaultdict
            from datetime import datetime
            
            influence = defaultdict(int)
            
            for i, (username, transcription, timestamp) in enumerate(messages):
                # Give points for starting conversations (long gap before)
                if i < len(messages) - 1:
                    next_time = datetime.strptime(messages[i+1][2], '%Y-%m-%d %H:%M:%S')
                    curr_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    gap = (curr_time - next_time).total_seconds()
                    
                    if gap > 120:  # Started new topic after 2min silence
                        influence[username] += 5
                
                # Give points for getting responses
                if i > 0 and messages[i-1][0] != username:
                    influence[username] += 2
                
                # Give points for long messages (more substance)
                word_count = len(transcription.split())
                if word_count > 15:
                    influence[username] += 3
            
            if not influence:
                return "Not enough conversation data"
            
            top_influencers = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = "👑 Most influential (conversation drivers):\n"
            for user, score in top_influencers:
                result += f"  {user}: {score} influence points\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Influence error: {e}")
            return "Error calculating influence"
    
    # Cringe detector
    if "cringe" in text_lower or "awkward" in text_lower or "uncomfortable" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute("""
                SELECT username, transcription, timestamp FROM transcriptions
                WHERE guild_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (str(guild_id),))
            
            messages = c.fetchall()
            conn.close()
            
            from datetime import datetime
            
            cringe_moments = []
            
            for i in range(len(messages) - 1):
                username, transcription, timestamp = messages[i]
                next_username, next_transcription, next_timestamp = messages[i+1]
                
                curr_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                next_time = datetime.strptime(next_timestamp, '%Y-%m-%d %H:%M:%S')
                silence = (curr_time - next_time).total_seconds()
                
                # Detect awkward silence (>8 seconds after certain phrases)
                cringe_words = ['actually', 'well technically', 'fun fact', 'to be fair', 'not to brag']
                if any(phrase in transcription.lower() for phrase in cringe_words) and silence > 8:
                    cringe_moments.append((username, transcription[:50], silence))
            
            if not cringe_moments:
                return "😌 No cringe detected recently"
            
            result = "😬 Recent cringe moments:\n"
            for user, text, silence in cringe_moments[:3]:
                result += f"  {user}: '{text}...' ({silence:.0f}s awkward silence)\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Cringe error: {e}")
            return "Error detecting cringe"
    
    # Argument predictor
    if "argument predict" in text_lower or "will argue" in text_lower or "fight predict" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get recent argument history between people
            c.execute("""
                SELECT username, transcription FROM transcriptions
                WHERE guild_id = ?
                AND (transcription LIKE '%no%' OR transcription LIKE '%wrong%' 
                     OR transcription LIKE '%actually%' OR transcription LIKE '%but%')
                ORDER BY timestamp DESC
                LIMIT 200
            """, (str(guild_id),))
            
            arguments = c.fetchall()
            conn.close()
            
            from collections import defaultdict
            
            # Count arguments between pairs
            argument_pairs = defaultdict(int)
            last_arguer = None
            
            for username, transcription in arguments:
                if last_arguer and last_arguer != username:
                    pair = tuple(sorted([username, last_arguer]))
                    argument_pairs[pair] += 1
                last_arguer = username
            
            if not argument_pairs:
                return "No argument patterns detected"
            
            # Find most argumentative pair
            top_pair = max(argument_pairs.items(), key=lambda x: x[1])
            person1, person2 = top_pair[0]
            count = top_pair[1]
            
            # Calculate probability (capped at 95%)
            probability = min(95, 30 + (count * 5))
            
            return f"🔮 {probability}% chance {person1} and {person2} argue in next 10 min (based on {count} past arguments)"
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error predicting arguments"
    
    # Accent challenge judge
    if "accent" in text_lower and ("judge" in text_lower or "challenge" in text_lower or "rate" in text_lower):
        try:
            # Use AI to judge the last thing said
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""
                SELECT username, transcription FROM transcriptions
                WHERE guild_id = ? AND username = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (str(guild_id), requestor.name))
            
            last_msg = c.fetchone()
            conn.close()
            
            if not last_msg:
                return "Say something first, then ask me to judge your accent"
            
            import httpx
            judge_prompt = f"Judge this attempt at doing an accent (be funny and harsh): '{last_msg[1]}'. Give a score out of 10 and roast them in ONE sentence (15 words max)."
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": judge_prompt, "stream": False}
                )
            
            if response.status_code == 200:
                judgment = response.json().get('response', '').strip()
                return f"🎭 Accent judge: {judgment}"
            
            return "Error judging accent"
            
        except Exception as e:
            print(f"Accent judge error: {e}")
            return "Error judging"
    
    # Life advice based on patterns
    if "life advice" in text_lower or "wisdom" in text_lower or "advice for me" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get user's recent messages
            c.execute("""
                SELECT transcription FROM transcriptions
                WHERE guild_id = ? AND username = ?
                ORDER BY timestamp DESC LIMIT 50
            """, (str(guild_id), requestor.name))
            
            messages = c.fetchall()
            conn.close()
            
            if len(messages) < 10:
                return "I need more data about you first"
            
            # Analyze patterns
            all_text = " ".join([m[0] for m in messages])
            
            import httpx
            advice_prompt = f"Based on these recent messages from a user, give them ONE piece of sarcastic but genuinely helpful life advice (20 words max): {all_text[:500]}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "gemma2:27b", "prompt": advice_prompt, "stream": False}
                )
            
            if response.status_code == 200:
                advice = response.json().get('response', '').strip()
                return f"💡 Life advice: {advice}"
            
            return "Unable to generate wisdom"
            
        except Exception as e:
            print(f"Advice error: {e}")
            return "Error generating advice"
    
    # Debate tracker - who wins arguments
    if "debate" in text_lower or "who wins arguments" in text_lower or "argument tracker" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find argument indicators (disagreement words followed by back-and-forth)
            c.execute("""
                SELECT username, transcription, timestamp FROM transcriptions
                WHERE guild_id = ? 
                AND (transcription LIKE '%no %' OR transcription LIKE '%wrong%' 
                     OR transcription LIKE '%actually%' OR transcription LIKE '%but %'
                     OR transcription LIKE '%disagree%' OR transcription LIKE '%nah%')
                ORDER BY timestamp DESC
                LIMIT 200
            """, (str(guild_id),))
            
            debates = c.fetchall()
            conn.close()
            
            if len(debates) < 5:
                return "Not enough argument data"
            
            # Track who gets the last word in arguments
            # An "argument win" = you spoke last after a disagreement word
            from collections import defaultdict
            debate_wins = defaultdict(int)
            last_debater = None
            
            for username, transcription, timestamp in reversed(debates):
                if any(word in transcription.lower() for word in ['no', 'wrong', 'actually', 'but', 'disagree', 'nah']):
                    if last_debater and last_debater != username:
                        # Previous person got the last word
                        debate_wins[last_debater] += 1
                    last_debater = username
            
            if not debate_wins:
                return "No clear debate winners detected"
            
            # Sort by wins
            top_debaters = sorted(debate_wins.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = "Debate champions (most last words):\n"
            for debater, wins in top_debaters:
                result += f"  {debater}: {wins} arguments won\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Debate tracker error: {e}")
            return "Error tracking debates"
    
    # Vocabulary analysis
    if "vocabulary" in text_lower or "reading level" in text_lower or "word stats" in text_lower:
        try:
            # Determine target user
            target_user = requestor.name
            if "my" not in text_lower:
                for member in guild.members:
                    if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                        target_user = member.name
                        break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get all messages from target user
            c.execute("""
                SELECT transcription FROM transcriptions
                WHERE guild_id = ? AND username = ?
            """, (str(guild_id), target_user))
            messages = c.fetchall()
            conn.close()
            
            if not messages:
                return f"No data for {target_user}"
            
            # Analyze vocabulary
            all_words = []
            for msg in messages:
                words = msg[0].lower().split()
                all_words.extend([w.strip('.,!?;:') for w in words if len(w) > 2])
            
            if not all_words:
                return f"Not enough words to analyze"
            
            total_words = len(all_words)
            unique_words = len(set(all_words))
            vocab_diversity = (unique_words / total_words * 100) if total_words > 0 else 0
            avg_word_length = sum(len(w) for w in all_words) / len(all_words)
            
            # Estimate reading level (Flesch-Kincaid approximation)
            # Simple version: longer words = higher level
            if avg_word_length < 4:
                reading_level = "Elementary (3rd-5th grade)"
            elif avg_word_length < 5:
                reading_level = "Middle School (6th-8th grade)"
            elif avg_word_length < 6:
                reading_level = "High School (9th-12th grade)"
            else:
                reading_level = "College level"
            
            return f"{target_user}: {unique_words:,} unique words out of {total_words:,} total ({vocab_diversity:.1f}% diversity). Avg word length: {avg_word_length:.1f}. Reading level: {reading_level}"
            
        except Exception as e:
            print(f"Vocabulary error: {e}")
            return "Error analyzing vocabulary"
    
    # Relationship graph - who talks to who
    if "relationship" in text_lower or "who talks to who" in text_lower or "conversation map" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get all users
            c.execute("""
                SELECT DISTINCT username FROM transcriptions 
                WHERE guild_id = ? 
                ORDER BY timestamp DESC LIMIT 100
            """, (str(guild_id),))
            users = [row[0] for row in c.fetchall()]
            
            if len(users) < 2:
                return "Need at least 2 people"
            
            # Count conversations between pairs
            pairs = {}
            for i, user1 in enumerate(users):
                for user2 in users[i+1:]:
                    # Count messages where both talked within 2 minutes of each other
                    c.execute("""
                        WITH user1_msgs AS (
                            SELECT timestamp FROM transcriptions 
                            WHERE guild_id = ? AND username = ?
                        ),
                        user2_msgs AS (
                            SELECT timestamp FROM transcriptions 
                            WHERE guild_id = ? AND username = ?
                        )
                        SELECT COUNT(*) FROM user1_msgs u1
                        JOIN user2_msgs u2 
                        ON ABS(julianday(u1.timestamp) - julianday(u2.timestamp)) * 1440 < 2
                    """, (str(guild_id), user1, str(guild_id), user2))
                    
                    count = c.fetchone()[0]
                    if count > 5:  # Only show meaningful relationships
                        pairs[f"{user1} ↔ {user2}"] = count
            
            conn.close()
            
            if not pairs:
                return "Not enough conversation data yet"
            
            # Sort by conversation count
            top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = "Top conversation pairs:\n"
            for pair, count in top_pairs:
                result += f"  {pair}: {count} interactions\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"Relationship error: {e}")
            return "Error analyzing relationships"
    
    # Energy level detection
    if "energy" in text_lower or "vibe check" in text_lower or "how's the vibe" in text_lower:
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get last 20 messages
            c.execute("""
                SELECT transcription, timestamp FROM transcriptions
                WHERE guild_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 20
            """, (str(guild_id),))
            recent = c.fetchall()
            conn.close()
            
            if len(recent) < 5:
                return "Not enough data yet"
            
            # Calculate energy indicators
            total_words = sum(len(msg[0].split()) for msg in recent)
            avg_words = total_words / len(recent)
            
            # Check for high energy indicators
            high_energy = sum(1 for msg in recent if any(word in msg[0].lower() for word in 
                ['!', 'lol', 'haha', 'lmao', 'fuck', 'shit', 'damn', 'wow', 'bruh']))
            
            # Calculate messages per minute (last 10 messages)
            if len(recent) >= 10:
                from datetime import datetime
                first = datetime.strptime(recent[9][1], '%Y-%m-%d %H:%M:%S')
                last = datetime.strptime(recent[0][1], '%Y-%m-%d %H:%M:%S')
                time_diff = (last - first).total_seconds() / 60
                msg_rate = 10 / time_diff if time_diff > 0 else 0
            else:
                msg_rate = 0
            
            # Determine energy level
            energy_score = (avg_words * 0.3) + (high_energy * 2) + (msg_rate * 3)
            
            if energy_score > 25:
                return f"Energy: 🔥 HIGH ENERGY ({int(energy_score)}%) - Chat is poppin!"
            elif energy_score > 15:
                return f"Energy: ⚡ Medium energy ({int(energy_score)}%) - Pretty active"
            elif energy_score > 5:
                return f"Energy: 😴 Low energy ({int(energy_score)}%) - Kinda dead"
            else:
                return f"Energy: 💀 Dead chat ({int(energy_score)}%)"
                
        except Exception as e:
            print(f"Energy error: {e}")
            return "Error checking energy"
    
    # Learning from corrections
    if any(phrase in text_lower for phrase in ['no jarvis', 'jarvis no', 'i meant', 'actually', 'correction']):
        try:
            # Get the last thing Jarvis said/transcribed
            conn = get_db_connection()
            c = conn.cursor()
            
            # Get what user is correcting to
            correction = text.replace('no jarvis', '').replace('jarvis no', '').replace('i meant', '').replace('actually', '').replace('correction', '').strip().lstrip(',').strip()
            
            if correction:
                # Store the correction
                c.execute("""CREATE TABLE IF NOT EXISTS corrections 
                            (guild_id TEXT, user_id TEXT, username TEXT, correction TEXT, 
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
                c.execute("""INSERT INTO corrections (guild_id, user_id, username, correction) 
                            VALUES (?, ?, ?, ?)""", 
                         (str(guild_id), str(requestor.id), requestor.name, correction))
                conn.commit()
                conn.close()
                
                return "Got it, I'll remember that."
        except Exception as e:
            print(f"Correction error: {e}")
            return "Noted."
    
    # Math Calculator
    if any(phrase in text_lower for phrase in ["what's", "whats", "calculate", "how much is"]) and any(op in text for op in ['+', '-', '*', '/', '%', 'percent']):
        try:
            import re
            # Extract math expression
            expr = re.sub(r'what\'s|whats|calculate|how much is|jarvis|,', '', text_lower).strip()
            
            # Handle percentages
            if 'percent of' in expr or '% of' in expr:
                parts = re.findall(r'(\d+\.?\d*)', expr)
                if len(parts) >= 2:
                    percent = float(parts[0])
                    number = float(parts[1])
                    result = (percent / 100) * number
                    return f"{percent}% of {number} is {result}"
            
            # Safe eval for basic math
            allowed = set('0123456789+-*/() .')
            if all(c in allowed for c in expr.replace(' ', '')):
                result = eval(expr)
                return f"{expr} equals {result}"
                
            return "Can't calculate that"
        except Exception as e:
            print(f"Math error: {e}")
            return "Math error"
    
    # Unit Conversions
    if "to" in text_lower and any(unit in text_lower for unit in ["celsius", "fahrenheit", "miles", "kilometers", "pounds", "kilos", "feet", "meters"]):
        try:
            import re
            numbers = re.findall(r'(\d+\.?\d*)', text_lower)
            if numbers:
                value = float(numbers[0])
                
                if "celsius" in text_lower and "fahrenheit" in text_lower:
                    result = (value * 9/5) + 32
                    return f"{value} celsius is {result:.1f} fahrenheit"
                elif "fahrenheit" in text_lower and "celsius" in text_lower:
                    result = (value - 32) * 5/9
                    return f"{value} fahrenheit is {result:.1f} celsius"
                elif ("miles" in text_lower or "mi" in text_lower) and ("kilometers" in text_lower or "km" in text_lower):
                    result = value * 1.60934
                    return f"{value} miles is {result:.1f} kilometers"
                elif ("kilometers" in text_lower or "km" in text_lower) and ("miles" in text_lower or "mi" in text_lower):
                    result = value * 0.621371
                    return f"{value} kilometers is {result:.1f} miles"
                elif "pounds" in text_lower and ("kilos" in text_lower or "kg" in text_lower):
                    result = value * 0.453592
                    return f"{value} pounds is {result:.1f} kilograms"
                elif ("kilos" in text_lower or "kg" in text_lower) and "pounds" in text_lower:
                    result = value * 2.20462
                    return f"{value} kilograms is {result:.1f} pounds"
                elif "feet" in text_lower and "meters" in text_lower:
                    result = value * 0.3048
                    return f"{value} feet is {result:.1f} meters"
                elif "meters" in text_lower and "feet" in text_lower:
                    result = value * 3.28084
                    return f"{value} meters is {result:.1f} feet"
                    
            return "Couldn't parse conversion"
        except Exception as e:
            print(f"Conversion error: {e}")
            return "Conversion error"
    
    # Voice Channel History
    if "who was here" in text_lower or "who was in voice" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Determine time range
            hours_back = 24  # Default to last 24 hours
            if "last night" in text_lower:
                hours_back = 12
            elif "yesterday" in text_lower:
                hours_back = 48
            elif "this week" in text_lower:
                hours_back = 168
                
            cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT DISTINCT username FROM transcriptions 
                        WHERE guild_id = ? AND timestamp > ?
                        ORDER BY timestamp DESC""",
                     (str(guild_id), cutoff_time))
            results = c.fetchall()
            conn.close()
            
            if results:
                users = [r[0] for r in results[:15]]  # Limit to 15 users
                timeframe = "last night" if hours_back == 12 else f"last {hours_back} hours"
                return f"In voice {timeframe}: {', '.join(users)}"
            return f"No voice activity in the last {hours_back} hours"
        except Exception as e:
            print(f"Voice history error: {e}")
            return "Error checking history"
    
    # Memory Search Engine
    if "when did" in text_lower or "search for" in text_lower or "find when" in text_lower or "show me when" in text_lower:
        try:
            # Extract search query
            search_term = None
            if "mention" in text_lower:
                # "when did someone mention pizza"
                search_term = text_lower.split("mention", 1)[1].strip()
            elif "talk about" in text_lower:
                # "when did alex talk about crypto"
                search_term = text_lower.split("talk about", 1)[1].strip()
            elif "say" in text_lower and "said" not in text_lower:
                # "when did someone say banana"
                search_term = text_lower.split("say", 1)[1].strip()
            elif "search for" in text_lower:
                search_term = text_lower.split("search for", 1)[1].strip()
            
            if not search_term or len(search_term) < 3:
                return "What should I search for?"
            
            # Clean up search term
            search_term = search_term.replace("?", "").replace(".", "").strip()
            
            # Check if searching for specific user
            target_user = None
            for member in guild.members:
                if member.name.lower() in text_lower or member.display_name.lower() in text_lower:
                    target_user = member.display_name
                    break
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Build query
            if target_user:
                c.execute("""SELECT username, transcription, timestamp 
                            FROM transcriptions 
                            WHERE guild_id = ? AND username = ? AND LOWER(transcription) LIKE ?
                            ORDER BY timestamp DESC LIMIT 10""",
                         (str(guild_id), target_user, f"%{search_term}%"))
            else:
                c.execute("""SELECT username, transcription, timestamp 
                            FROM transcriptions 
                            WHERE guild_id = ? AND LOWER(transcription) LIKE ?
                            ORDER BY timestamp DESC LIMIT 10""",
                         (str(guild_id), f"%{search_term}%"))
            
            results = c.fetchall()
            conn.close()
            
            if results:
                from datetime import datetime
                
                # Format results
                if len(results) == 1:
                    r = results[0]
                    dt = datetime.fromisoformat(r[2])
                    time_ago = datetime.now() - dt
                    
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days} days ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600} hours ago"
                    else:
                        time_str = f"{time_ago.seconds // 60} minutes ago"
                    
                    return f"{r[0]} mentioned it {time_str}: '{r[1][:100]}'"
                else:
                    # Multiple results
                    most_recent = results[0]
                    dt = datetime.fromisoformat(most_recent[2])
                    time_ago = datetime.now() - dt
                    
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days} days ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600} hours ago"
                    else:
                        time_str = f"{time_ago.seconds // 60} minutes ago"
                    
                    return f"Found {len(results)} mentions. Most recent: {most_recent[0]} said it {time_str}"
            
            return f"No one has mentioned '{search_term}' in recorded history"
            
        except Exception as e:
            print(f"Memory search error: {e}")
            return "Search error"
    
    # Auto DJ Controls
    if "enable auto dj" in text_lower or "turn on auto dj" in text_lower or "start auto dj" in text_lower:
        AUTO_DJ_ENABLED[guild_id] = True
        # Start monitoring task
        bot.loop.create_task(auto_dj_monitor(guild_id))
        return "Auto DJ enabled. I'll play music based on conversation energy."
    
    if "disable auto dj" in text_lower or "turn off auto dj" in text_lower or "stop auto dj" in text_lower:
        AUTO_DJ_ENABLED[guild_id] = False
        return "Auto DJ disabled."
    
    if "conversation energy" in text_lower or "energy level" in text_lower or "vibe check" in text_lower:
        energy = await analyze_conversation_energy(guild_id)
        if energy > 70:
            vibe = "high energy, hyped up"
        elif energy > 40:
            vibe = "moderate energy, engaged"
        elif energy > 15:
            vibe = "low energy, chill"
        else:
            vibe = "very quiet, almost dead"
        return f"Current conversation energy: {energy}/100. Vibe is {vibe}."
    
    # Fact Checker Controls
    if "enable fact checker" in text_lower or "turn on fact checker" in text_lower or "start fact checking" in text_lower:
        FACT_CHECKER_ENABLED[guild_id] = True
        # Start monitoring task
        bot.loop.create_task(fact_checker_monitor(guild_id))
        return "Fact checker enabled. I'll call out false claims in real-time."
    
    if "disable fact checker" in text_lower or "turn off fact checker" in text_lower or "stop fact checking" in text_lower or "disable fact check" in text_lower:
        FACT_CHECKER_ENABLED[guild_id] = False
        return "Fact checker disabled. People can lie freely now."
    
    # Psychological Warfare Controls
    if "disable gaslighting detector" in text_lower or "turn off gaslighting detector" in text_lower:
        
        return "Gaslighting detector disabled. People can gaslight freely now."
    
    if "disable manipulation tracker" in text_lower or "turn off manipulation tracker" in text_lower:
        MANIPULATION_TRACKER_ENABLED[guild_id] = False
        
    
    if "enable psychological warfare" in text_lower or "full warfare mode" in text_lower:
        GASLIGHTING_DETECTOR_ENABLED[guild_id] = True
        MANIPULATION_TRACKER_ENABLED[guild_id] = True
        bot.loop.create_task(manipulation_monitor(guild_id))
        return "PSYCHOLOGICAL WARFARE MODE ACTIVATED. All manipulation tactics will be exposed. Friendships may be lost."
    
    # Relationship Destroyer Commands
    if "is he lying" in text_lower or "is she lying" in text_lower or "lie detector" in text_lower:
        # This will be called automatically via voice biometrics
        # But we can provide manual check
        return "Voice stress analysis is automatic. Watch for alerts."
    
    if "best one liner" in text_lower or "funniest one liner" in text_lower or "top one liner" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            # Analyze last 30 days
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            
            conn = get_db_connection()
            c = conn.cursor()
            
            # Find short, punchy messages (5-15 words)
            c.execute("""SELECT username, transcription, timestamp 
                        FROM transcriptions 
                        WHERE guild_id = ? AND timestamp > ?
                        ORDER BY timestamp DESC LIMIT 500""",
                     (str(guild_id), cutoff))
            
            results = c.fetchall()
            conn.close()
            
            if not results:
                return "No data yet"
            
            # Score one-liners
            oneliners = []
            for username, text, timestamp in results:
                word_count = len(text.split())
                
                # Must be 5-15 words
                if 5 <= word_count <= 15:
                    score = 0
                    text_lower = text.lower()
                    
                    # Bonus for ending with punctuation
                    if text.endswith(('!', '?', '.')):
                        score += 10
                    
                    # Bonus for roast indicators
                    roast_words = ['shit', 'fuck', 'damn', 'hell', 'ass', 'bitch']
                    score += sum(5 for word in roast_words if word in text_lower)
                    
                    # Bonus for wit indicators
                    wit_words = ['actually', 'technically', 'literally', 'imagine', 'meanwhile']
                    score += sum(3 for word in wit_words if word in text_lower)
                    
                    # Bonus for question marks (sarcasm)
                    score += text.count('?') * 5
                    
                    # Bonus for caps words (emphasis)
                    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
                    score += len(caps_words) * 3
                    
                    # Penalty for being too common
                    common_phrases = ['i know', 'yeah', 'okay', 'alright', 'whatever']
                    if any(phrase in text_lower for phrase in common_phrases):
                        score -= 10
                    
                    oneliners.append((username, text, score))
            
            if not oneliners:
                return "No good one-liners found"
            
            # Sort by score
            top_oneliners = sorted(oneliners, key=lambda x: x[2], reverse=True)[:5]
            
            # Format response
            result = "Top one-liners: "
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            
            winners = []
            for i, (username, text, score) in enumerate(top_oneliners):
                winners.append(f"{medals[i]} {username}: '{text}'")
            
            return result + " | ".join(winners)
            
        except Exception as e:
            print(f"One-liner error: {e}")
            return "Error finding one-liners"
    
    # Activity Search Commands
    if "show messages from" in text_lower or "what did" in text_lower and "say in" in text_lower:
        try:
            # Extract username
            target_user = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target_user = member.display_name
                    break
            
            if not target_user:
                return "Who do you want to check?"
            
            # Get time filter
            from datetime import datetime, timedelta
            if "today" in text_lower:
                cutoff = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            elif "yesterday" in text_lower:
                cutoff = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
            else:
                cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT message_content, timestamp FROM text_messages 
                        WHERE guild_id = ? AND username = ? AND timestamp > ?
                        ORDER BY timestamp DESC LIMIT 10""",
                     (str(guild_id), target_user, cutoff))
            results = c.fetchall()
            conn.close()
            
            if results:
                count = len(results)
                return f"{target_user} sent {count} messages. Most recent: '{results[0][0][:80]}'"
            return f"No messages from {target_user} in that timeframe"
        except Exception as e:
            print(f"Message search error: {e}")
            return "Error searching messages"
    
    if "who joined voice" in text_lower or "who's been in voice" in text_lower or "voice activity" in text_lower:
        try:
            from datetime import datetime, timedelta
            
            if "today" in text_lower:
                cutoff = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            else:
                cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT username, channel_name, action, timestamp FROM voice_activity 
                        WHERE guild_id = ? AND timestamp > ? AND action = 'JOIN'
                        ORDER BY timestamp DESC LIMIT 10""",
                     (str(guild_id), cutoff))
            results = c.fetchall()
            conn.close()
            
            if results:
                unique_users = set(r[0] for r in results)
                return f"{len(unique_users)} people joined voice recently: {', '.join(list(unique_users)[:5])}"
            return "No voice activity recently"
        except Exception as e:
            print(f"Voice activity search error: {e}")
            return "Error checking voice activity"
    
    if "show attachments from" in text_lower or "what files did" in text_lower:
        try:
            target_user = None
            for member in guild.members:
                if member.display_name.lower() in text_lower or member.name.lower() in text_lower:
                    target_user = member.display_name
                    break
            
            if not target_user:
                return "Who's attachments do you want to see?"
            
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""SELECT attachment_urls, timestamp FROM text_messages 
                        WHERE guild_id = ? AND username = ? AND has_attachment = 1 AND timestamp > ?
                        ORDER BY timestamp DESC LIMIT 5""",
                     (str(guild_id), target_user, cutoff))
            results = c.fetchall()
            conn.close()
            
            if results:
                return f"{target_user} posted {len(results)} attachments in the last week"
            return f"{target_user} hasn't posted any files recently"
        except Exception as e:
            print(f"Attachment search error: {e}")
            return "Error searching attachments"
    
    # APOCALYPSE FEATURES - Friendship Destroyers
    
    # Behavioral Pattern Lock

# Transcription loop
async def transcribe_loop(guild_id, sink):
    global transcription_language
    await asyncio.sleep(2)
    
    while guild_id in voice_clients:
        try:
            # Process each user's buffer once
            users_to_process = list(sink.user_buffers.keys())
            
            for user_id in users_to_process:
                if user_id not in sink.user_buffers:
                    continue
                    
                data = sink.user_buffers[user_id]
                buffer = data['buffer']
                user = data['user']
                
                if len(buffer) > 3:
                    audio_data = b''.join(buffer)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # INSTANT WAKE WORD DETECTION with Porcupine
                    if porcupine is not None:
                        try:
                            # Resample to 16kHz for Porcupine
                            if len(audio_array) % 2 == 0:
                                audio_mono = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                            else:
                                audio_mono = audio_array
                            
                            # Porcupine expects specific frame length
                            frame_length = porcupine.frame_length
                            num_frames = len(audio_mono) // frame_length
                            
                            for i in range(num_frames):
                                frame = audio_mono[i * frame_length:(i + 1) * frame_length]
                                keyword_index = porcupine.process(frame)
                                
                                if keyword_index >= 0:
                                    # JARVIS DETECTED - INSTANT BEEP!
                                    print(f"🎯 JARVIS detected instantly!")
                                    if guild_id in voice_clients and 'vc' in voice_clients[guild_id]:
                                        vc = voice_clients[guild_id]['vc']
                                        # Play instant click
                                        try:
                                            audio_source = discord.FFmpegPCMAudio("/tmp/beep_click.wav")
                                            vc.play(audio_source, after=lambda e: None)
                                        except:
                                            pass
                                    break
                        except Exception as e:
                            pass
                    
                    # Clear buffer immediately to prevent reprocessing
                    sink.user_buffers[user_id]['buffer'] = []
                    
                    if len(audio_array) > 48000:
                        if len(audio_array) % 2 == 0:
                            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        audio_16k = librosa.resample(audio_float, orig_sr=48000, target_sr=16000)
                        
                        if len(audio_16k) > 16000:
                            # VOICE BIOMETRICS ANALYSIS
                            voice_features = analyze_voice_features(audio_array)
                            if voice_features:
                                # Detect emotion
                                emotion = detect_emotion_from_voice(voice_features)
                                
                                # Load voice profiles
                                profiles = load_voice_profiles()
                                user_id = str(user.id)
                                
                                # Create or update baseline
                                if user_id not in profiles:
                                    profiles[user_id] = {
                                        'username': user.display_name,
                                        'baseline': voice_features,
                                        'samples': 1
                                    }
                                else:
                                    # Update baseline (running average)
                                    baseline = profiles[user_id]['baseline']
                                    samples = profiles[user_id]['samples']
                                    for key in voice_features:
                                        baseline[key] = (baseline[key] * samples + voice_features[key]) / (samples + 1)
                                    profiles[user_id]['samples'] = samples + 1
                                
                                # Check for intoxication
                                is_intoxicated, intox_score = detect_intoxication(
                                    voice_features, 
                                    profiles[user_id]['baseline']
                                )
                                
                                # Check for lying (voice stress)
                                is_lying, lie_confidence = await detect_voice_stress_lying(
                                    voice_features,
                                    profiles[user_id]['baseline']
                                )
                                
                                if is_lying and lie_confidence > 60:
                                    print(f"🤥 {user.display_name} shows voice stress (lying confidence: {lie_confidence}%)")
                                
                                # Save profiles
                                save_voice_profiles(profiles)
                                
                                # Log interesting findings
                                if emotion not in ['calm', 'neutral']:
                                    print(f"🎭 {user.display_name} sounds {emotion}")
                                if is_intoxicated:
                                    print(f"🍺 {user.display_name} might be intoxicated (score: {intox_score})")
                            
                            lang = transcription_language.get(guild_id, "en")
                            segments, info = whisper_model.transcribe(
                                audio_16k,
                                language=lang,
                                beam_size=3,
                                vad_filter=True,
                                vad_parameters=dict(
                                    threshold=0.4,
                                    min_speech_duration_ms=500,
                                    min_silence_duration_ms=4000
                                )
                            )
                            torch.cuda.empty_cache()  # Free VRAM after Whisper
                            
                            transcription = " ".join([segment.text.strip() for segment in segments])
                            if transcription and len(transcription) > 1:
                                username = user.display_name
                                
                                conn = get_db_connection()
                                c = conn.cursor()
                                c.execute("""CREATE TABLE IF NOT EXISTS transcriptions 
                                            (user_id TEXT, username TEXT, guild_id TEXT, channel_id TEXT, 
                                            transcription TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
                                c.execute("""INSERT INTO transcriptions 
                                            (user_id, username, guild_id, channel_id, transcription)
                                            VALUES (?, ?, ?, ?, ?)""",
                                         (str(user_id), username, str(guild_id), str(sink.channel_id), transcription))
                                conn.commit()
                                conn.close()
                                
                                print(f"✅ [{username}]: {transcription}")

                                # Clear buffer after every transcription
                                # Highlight detection - rapid conversation = funny moment
                                if guild_id not in globals():
                                    globals()['last_transcript_times'] = {}
                                
                                if guild_id not in globals()['last_transcript_times']:
                                    globals()['last_transcript_times'][guild_id] = []
                                
                                # Track recent message times
                                now = datetime.now()
                                globals()['last_transcript_times'][guild_id].append(now)
                                
                                # Keep only last 10 seconds
                                globals()['last_transcript_times'][guild_id] = [
                                    t for t in globals()['last_transcript_times'][guild_id] 
                                    if (now - t).total_seconds() < 10
                                ]
                                
                                # If 5+ messages in 10 seconds = highlight moment
                                if len(globals()['last_transcript_times'][guild_id]) >= 5:
                                    # Save highlight
                                    conn_hl = get_db_connection()
                                    c_hl = conn_hl.cursor()
                                    c_hl.execute("""CREATE TABLE IF NOT EXISTS highlights 
                                                   (guild_id TEXT, timestamp DATETIME, context TEXT)""")
                                    
                                    # Get last 10 messages as context
                                    c_hl.execute("""SELECT username, transcription FROM transcriptions 
                                                   WHERE guild_id = ? ORDER BY timestamp DESC LIMIT 10""",
                                                (str(guild_id),))
                                    context_msgs = c_hl.fetchall()
                                    context = "\n".join([f"{m[0]}: {m[1]}" for m in reversed(context_msgs)])
                                    
                                    c_hl.execute("""INSERT INTO highlights (guild_id, timestamp, context) 
                                                   VALUES (?, ?, ?)""",
                                                (str(guild_id), now.strftime('%Y-%m-%d %H:%M:%S'), context))
                                    conn_hl.commit()
                                    conn_hl.close()
                                    
                                    print(f"🎬 Highlight detected at {now}")
                                    # Clear to avoid duplicate saves
                                    globals()['last_transcript_times'][guild_id] = []

                                
                                # Wake word detection
                                lower_text = transcription.lower()
                                # Check for wake word (jarvis or travis anywhere in text)
                                if any(wake in lower_text for wake in ['jarvis', 'travis']):
                                    # Wait for complete command (give user time to finish speaking)
                                    await asyncio.sleep(2.5)

                                    vc = voice_clients[guild_id]['vc']
                                    guild_obj = bot.get_guild(guild_id)
                                    
                                    clean_message = transcription.lower()
                                    for wake in ['hey jarvis', 'hi jarvis', 'okay jarvis', 'ok jarvis', 'jarvis', 'travis', 'hey', 'hi', 'okay', 'ok']:
                                        clean_message = clean_message.replace(wake, '').strip()
                                    clean_message = clean_message.lstrip(',').strip()
                                    
                                    if len(clean_message) > 2:
                                        user_question = clean_message
                                    else:
                                        user_question = "hello"
                                    
                                    # IMMEDIATE ACKNOWLEDGMENT BEEP
                                    try:
                                        import numpy as beep_np
                                        import wave as beep_wave
                                        import tempfile as beep_temp
                                        
                                        # Generate sharp click acknowledgment
                                        sample_rate = 48000
                                        duration = 0.05  # Quick 50ms
                                        frequency = 2000  # High frequency
                                        t = beep_np.linspace(0, duration, int(sample_rate * duration))
                                        
                                        # Exponential decay for click sound
                                        envelope = beep_np.exp(-t * 50)
                                        
                                        # Generate click
                                        beep_data = (beep_np.sin(2 * beep_np.pi * frequency * t) * envelope * 0.3 * 32767).astype(beep_np.int16)
                                        
                                        with beep_temp.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                                            with beep_wave.open(fp.name, 'wb') as wav:
                                                wav.setnchannels(1)
                                                wav.setsampwidth(2)
                                                wav.setframerate(sample_rate)
                                                wav.writeframes(beep_data.tobytes())
                                            beep_file = fp.name
                                        
                                        # Play beep immediately
                                        audio_source = discord.FFmpegPCMAudio(beep_file)
                                        vc.play(audio_source, after=lambda e: None)
                                        await asyncio.sleep(0.2)  # Let beep play
                                    except Exception as e:
                                        import traceback
                                        print(f"Beep error: {e}")
                                        traceback.print_exc()
                                    
                                    print(f"🧠 Command: {user_question}")
                                    result = await handle_voice_command(user_question, guild_obj, user)
                                    if result:
                                        # Only use TTS if music isn't currently playing
                                        skip_tts_patterns = ['Now playing:', 'Added to queue:', 'Random:', 'Skipped.', 'Paused', 'Resumed', 'Playing:', 'Playing soundboard:']
                                        should_speak = not any(pattern in result for pattern in skip_tts_patterns)
                                        
                                        if should_speak and not vc.is_playing():
                                            await asyncio.to_thread(speak_text, result, vc, guild_id)
                                        print(f"✅ {result}")
                                
            
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(5)



async def monitor_opus_health():
    """Monitor for OpusError crashes and auto-reconnect"""
    await asyncio.sleep(30)
    
    while opus_monitoring_active:
        try:
            for guild_id, data in list(voice_clients.items()):
                vc = data.get('vc')
                if not vc or not vc.is_connected():
                    continue
                
                # Check if we're still receiving packets
                now = asyncio.get_event_loop().time()
                if guild_id in last_packet_time:
                    time_since_last = now - last_packet_time[guild_id]
                    
                    # If no packets for 45 seconds and users are in channel
                    humans = [m for m in vc.channel.members if not m.bot]
                    if time_since_last > 45 and len(humans) > 0:
                        print(f"⚠️ OpusError detected in {vc.channel.name}, reconnecting...")
                        
                        channel = vc.channel
                        
                        # Full reconnect
                        await vc.disconnect()
                        del voice_clients[guild_id]
                        await asyncio.sleep(3)
                        
                        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
                        await asyncio.sleep(1)
                        
                        sink = AudioSink(guild_id, channel.id)
                        vc.listen(sink)
                        
                        voice_clients[guild_id] = {'vc': vc, 'sink': sink}
                        conversation_history[guild_id] = []
                        bot.loop.create_task(transcribe_loop(guild_id, sink))
                        
                        last_packet_time[guild_id] = now
                        print(f"✅ Recovered from OpusError in {channel.name}")
            
            await asyncio.sleep(20)
        except Exception as e:
            print(f"Monitor error: {e}")
            await asyncio.sleep(5)

@bot.event
async def on_message(message):
    """Handle text commands and log all messages"""
    if message.author.bot:
        return
    
    # Log all text messages to database
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check for attachments/links and DOWNLOAD THEM
        has_attachment = len(message.attachments) > 0
        attachment_urls = []
        local_paths = []
        
        if has_attachment:
            from datetime import datetime
            import aiohttp
            
            date_folder = datetime.now().strftime("%Y-%m-%d")
            save_dir = f"{ATTACHMENTS_DIR}/{date_folder}"
            os.makedirs(save_dir, exist_ok=True)
            
            for att in message.attachments:
                try:
                    # Generate safe filename
                    timestamp = datetime.now().strftime("%H%M%S")
                    safe_filename = f"{message.author.display_name}_{timestamp}_{att.filename}"
                    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._- ")
                    local_path = f"{save_dir}/{safe_filename}"
                    
                    # Download file
                    async with aiohttp.ClientSession() as session:
                        async with session.get(att.url) as resp:
                            if resp.status == 200:
                                with open(local_path, 'wb') as f:
                                    f.write(await resp.read())
                                
                                attachment_urls.append(att.url)
                                local_paths.append(local_path)
                                
                                file_size = os.path.getsize(local_path) / 1024 / 1024  # MB
                                print(f"💾 Downloaded: {safe_filename} ({file_size:.2f}MB)")
                except Exception as e:
                    print(f"Download error for {att.filename}: {e}")
                    attachment_urls.append(att.url)
                    local_paths.append("FAILED")
        
        attachment_urls_str = ",".join(attachment_urls)
        local_paths_str = ",".join(local_paths)
        
        c.execute("""INSERT INTO text_messages 
                    (guild_id, channel_id, user_id, username, message_content, has_attachment, attachment_urls)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (str(message.guild.id) if message.guild else "DM",
                  str(message.channel.id),
                  str(message.author.id),
                  message.author.display_name,
                  message.content,
                  has_attachment,
                  f"{attachment_urls_str}|||{local_paths_str}"))
        conn.commit()
        conn.close()
        
        print(f"📝 [{message.author.display_name}] in #{message.channel.name}: {message.content[:50]}")
    except Exception as e:
        print(f"Message logging error: {e}")
    
    # !sorry command to unmute
    if message.content.lower() in ['!sorry', '!unmute', '!apologize']:
        guild_id = message.guild.id
        user_id = message.author.id
        
        if guild_id in muted_users and user_id in muted_users[guild_id]:
            muted_users[guild_id].remove(user_id)
            await message.channel.send(f"✅ {message.author.display_name} unmuted")
        else:
            await message.channel.send(f"{message.author.display_name} is not muted")
    
    await bot.process_commands(message)

@bot.event
async def on_voice_state_update(member, before, after):
    """Auto-rejoin when someone joins voice"""
    if member.bot:
        return
    
    # Someone joined a voice channel
    if after.channel and not before.channel:
        guild_id = after.channel.guild.id
        
# Bot not in voice for this guild
        if guild_id not in voice_clients or 'vc' not in voice_clients[guild_id]:
            print(f"🔔 {member.display_name} joined, auto-rejoining...")
            await asyncio.sleep(1)
            
            try:
                vc = await after.channel.connect(cls=voice_recv.VoiceRecvClient)
                sink = AudioSink(guild_id, after.channel.id)
                vc.listen(sink)
                
                voice_clients[guild_id] = {'vc': vc, 'sink': sink}
                conversation_history[guild_id] = []
                bot.loop.create_task(transcribe_loop(guild_id, sink))
                save_voice_state(guild_id, after.channel.id)
                
                print(f"✅ Auto-rejoined: {after.channel.name}")
            except Exception as e:
                print(f"❌ Auto-rejoin failed: {e}")
    
    # Someone left a voice channel
    elif before.channel and not after.channel:
        guild_id = before.channel.guild.id
        
        if guild_id in voice_clients and 'vc' in voice_clients[guild_id]:
            vc = voice_clients[guild_id]['vc']
            
            # Check if bot is in the channel they left from
            if vc.channel == before.channel:
                # Count humans left in channel
                humans = [m for m in vc.channel.members if not m.bot]
                
                if len(humans) == 0:
                    print(f"👋 Last person left, disconnecting from {before.channel.name}")
                    await vc.disconnect()
                    del voice_clients[guild_id]
                    if guild_id in conversation_history:
                        del conversation_history[guild_id]

# Database optimization
def optimize_database():
    """Create indexes for better performance"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Ensure tables exist before creating indexes
        c.execute("""CREATE TABLE IF NOT EXISTS user_context 
                    (guild_id TEXT, user_id TEXT, username TEXT, context_key TEXT, 
                    context_value TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS corrections 
                    (guild_id TEXT, user_id TEXT, username TEXT, correction TEXT, 
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS life_events 
                    (user_id TEXT, guild_id TEXT, username TEXT, event_type TEXT, 
                     event_description TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS bookmarks 
                    (guild_id TEXT, user_id TEXT, username TEXT, transcription TEXT, 
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS text_messages 
                    (guild_id TEXT, channel_id TEXT, user_id TEXT, username TEXT, 
                     message_content TEXT, has_attachment BOOLEAN, attachment_urls TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS voice_activity 
                    (guild_id TEXT, channel_id TEXT, channel_name TEXT, user_id TEXT, 
                     username TEXT, action TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        
        # Apocalypse feature tables
        
        
        
        
        
        
        
        
        
        
        
        
        # Create indexes on frequently queried columns
        c.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_guild_timestamp ON transcriptions(guild_id, timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_username ON transcriptions(username)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_transcriptions_guild_user ON transcriptions(guild_id, username)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_music_history_guild ON music_history(guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_music_history_song ON music_history(song_title)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_blacklisted_guild ON blacklisted_songs(guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_context_user ON user_context(user_id, guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_custom_commands_guild ON custom_commands(guild_id)")
        # Additional indexes for performance
        c.execute("CREATE INDEX IF NOT EXISTS idx_nicknames_user_guild ON nicknames(user_id, guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_corrections_guild ON corrections(guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_life_events_user ON life_events(user_id, guild_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bookmarks_guild ON bookmarks(guild_id)")
        
        conn.commit()
        conn.close()
        logger.info("Database optimized with indexes")
    except Exception as e:
        print(f"Database optimization error: {e}")

# Web Dashboard
async def dashboard_handler(request):
    """Serve web dashboard"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jarvis Bot Dashboard</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #1a1a1a; color: #fff; }
            .stat { background: #2a2a2a; padding: 15px; margin: 10px; border-radius: 5px; }
            h1 { color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>🤖 Jarvis Bot Dashboard</h1>
        <div class="stat">
            <h3>Status</h3>
            <p>Bot Version: """ + BOT_VERSION + """</p>
            <p>Active Servers: """ + str(len(voice_clients)) + """</p>
            <p>Debug Mode: """ + str(DEBUG_MODE) + """</p>
        </div>
        <div class="stat">
            <h3>Statistics</h3>
            <p>This is a basic dashboard. More features coming soon!</p>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def start_web_server():
    """Start web dashboard on port 9000"""
    try:
        app = web.Application()
        app.router.add_get('/', dashboard_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 9000)
        await site.start()
        logger.info("Web dashboard running at http://localhost:9000")
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.warning("Port 9000 already in use, web dashboard not started")
        else:
            logger.error(f"Web server error: {e}")

@bot.event  
async def on_ready():
    global bot_start_time
    from datetime import datetime
    bot_start_time = datetime.now()
    
    print(f'{bot.user} connected!')
    
    # Optimize database on startup
    optimize_database()
    
    # Start web dashboard
    bot.loop.create_task(start_web_server())
    saved = load_voice_states()
    for gid_str, cid_str in saved.items():
        try:
            guild = bot.get_guild(int(gid_str))
            if guild:
                channel = guild.get_channel(int(cid_str))
                if channel:
                    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
                    sink = AudioSink(int(gid_str), channel.id)
                    vc.listen(sink)
                    voice_clients[int(gid_str)] = {'vc': vc, 'sink': sink}
                    conversation_history[int(gid_str)] = []
                    bot.loop.create_task(transcribe_loop(int(gid_str), sink))
                    print(f"✅ Auto-rejoined: {channel.name}")
        except Exception as e:
            print(f"❌ Auto-rejoin failed: {e}")
    print('Ready!')
    bot.loop.create_task(monitor_opus_health())

@bot.command(name='join')
async def join(ctx):
    if not ctx.author.voice:
        await ctx.send("Join voice first!")
        return
    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id
    if guild_id in voice_clients:
        await ctx.send("Already listening!")
        return
    try:
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        sink = AudioSink(guild_id, channel.id)
        vc.listen(sink)
        voice_clients[guild_id] = {'vc': vc, 'sink': sink}
        conversation_history[guild_id] = []
        bot.loop.create_task(transcribe_loop(guild_id, sink))
        save_voice_state(guild_id, channel.id)
        await ctx.send("Listening! 🎤")
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command(name='leave')
async def leave(ctx):
    guild_id = ctx.guild.id
    if guild_id not in voice_clients:
        await ctx.send("Not in voice!")
        return
    try:
        await voice_clients[guild_id]['vc'].disconnect()
        del voice_clients[guild_id]
        await ctx.send("Left voice!")
    except Exception as e:
        await ctx.send(f"Error: {e}")

if __name__ == "__main__":
    import sys
    TOKEN = sys.argv[1] if len(sys.argv) > 1 else None
    if TOKEN:
        bot.run(TOKEN)

# Suppress noisy voice recv logs
import logging
logging.getLogger('discord.ext.voice_recv.reader').setLevel(logging.WARNING)
