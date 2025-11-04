# Web Chatbot Demo

## Quick Start

Run the web-based chatbot interface:

```bash
streamlit run code/web_chatbot.py
```

Or use the launcher:

```bash
python run_web_demo.py
```

The web interface will open at `http://localhost:8501`

## Features

- **Interactive Chat Interface**: Clean, modern chat UI
- **Real-time Answers**: Get instant responses from the RAG system
- **Source Citations**: See which documents were used for each answer
- **Sample Questions**: Quick access to common questions
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## For Presentations

This web interface is perfect for live demonstrations:

1. **Start the server** before your presentation
2. **Share the URL** with attendees (on same network)
3. **Attendees can ask questions** in real-time
4. **Show the interface** on your screen or projector

## Network Access

To allow attendees on the same network to access:

1. Find your IP address:
   ```bash
   # macOS/Linux
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. Share the URL: `http://YOUR_IP:8501`

3. Make sure firewall allows port 8501

## Troubleshooting

**Port already in use:**
```bash
streamlit run code/web_chatbot.py --server.port 8502
```

**API key not found:**
- Make sure `.env` file exists with `OPENROUTER_API_KEY` or `OPENAI_API_KEY`

**Vector store not found:**
- Run `python code/02_create_vectorstore.py` first

