#!/bin/bash
set -e

echo "🤖 JARVIS Lite Setup"
echo "===================="
echo ""

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Discord token!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Discord bot token"
echo "2. Make sure Ollama is running: ollama serve"
echo "3. Run bot: python bot_lite.py"
