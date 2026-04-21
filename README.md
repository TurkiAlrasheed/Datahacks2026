## RoboRanger

An agent that can identify the various species in the environment and can interact with the user to make ecological sciences more accessible.

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

### Future plans

power increase + LLM local + slm for self-driving + check SME top floor for designing wheels
