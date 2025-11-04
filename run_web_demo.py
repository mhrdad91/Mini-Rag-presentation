#!/usr/bin/env python3
"""
Quick launcher for the web chatbot demo
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit web chatbot."""
    script_path = Path(__file__).parent / "code" / "web_chatbot.py"
    
    print("=" * 80)
    print("Starting RAG Web Chatbot")
    print("=" * 80)
    print()
    print("The web interface will open in your browser.")
    print("If it doesn't open automatically, go to:")
    print("  http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 80)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)

if __name__ == "__main__":
    main()

