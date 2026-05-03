## RoboRanger

A portable AI agent that can identify the various species in your environment and give you a park ranger style tour guide of your surrounding ecosystem, making ecological sciences more interactive and accessible.

More details: https://devpost.com/software/roboranger?ref_content=my-projects-tab&ref_feature=my_projects

### Quick Start

**Terminal 1 — server:**

```bash
cd ~/Desktop/Datahacks2026/TourGuide_Agent
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — ngrok tunnel:**

```bash
ngrok http 8000
```

**Arduino UNO Q:**

```bash
sudo SERVER_URL=https://<ngrok-url> python3 arduino_client.py --camera 2 --mic-device hw:0,0
```

### Current setup

MobileNetV3-Large running on Arduino, with web scraping for more images if necessary (found in cnn/web-scraper.py)

must run build corpus before compile blurbs, because it tries to find the existing DB to add to it

### Future plans

power increase + LLM local + slm for self-driving + check SME top floor for designing wheels
