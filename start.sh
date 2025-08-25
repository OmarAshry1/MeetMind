#!/bin/bash

echo "🚀 Starting MeetMind Discord Bot..."

# Check if required environment variables are set
if [ -z "$DISCORD_TOKEN" ]; then
    echo "❌ ERROR: DISCORD_TOKEN environment variable is not set!"
    echo "Please set DISCORD_TOKEN in your Railway environment variables."
    exit 1
fi

echo "✅ Environment variables check passed"
echo "🌐 Starting Flask healthcheck server..."
echo "🤖 Starting Discord bot..."

# Start the bot
python MeetMind_local.py 