# ⚔️ Commander AI Arena

A browser-based Magic: The Gathering Commander application where you play your physical deck against AI opponents. Features a custom Python game engine, real-time card lookups via the Scryfall API, voice command support, and a visual board state. Currently uses a heuristic AI engine with a machine learning model in development as a replacement for the Claude API.

![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude_API-191919?style=flat&logo=anthropic&logoColor=white)

---

## ✨ Features

- **Custom Python Game Engine** — Full MTG rules enforcement including actions, effects, card parsing, and game state management written from scratch in Python
- **Heuristic AI Opponents** — Rule-based AI makes strategic decisions using a heuristic evaluation engine; ML model in development as a replacement
- **Claude API Integration** — Anthropic Claude API used for AI decision-making while the custom ML engine is completed
- **Scryfall API Integration** — Real-time card data, rulings, and artwork fetched directly from Scryfall's database
- **Voice Commands** — Web Speech API lets you speak actions instead of clicking
- **Visual Board State** — Interactive browser UI displays each player's board, hand, life total, and graveyard
- **Commander Format Rules** — Handles Commander damage, command zone, color identity, and multiplayer logic
- **Debug Panel** — Built-in debug overlay for monitoring AI turn state and diagnosing stalls
- **Abort Timeouts** — Prevents AI turns from hanging indefinitely with graceful error recovery

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | Node.js, Express |
| Game Engine | Python |
| AI (Current) | Heuristic engine + Anthropic Claude API |
| AI (In Development) | Custom Python ML Model |
| Card Data | Scryfall API |
| Voice | Web Speech API |

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.14+
- Anthropic API key

### Installation

```bash
git clone https://github.com/Crawv01/mtg-ai-commander.git

# Install Node dependencies
cd mtg-ai-commander/commander-arena/server
npm install

# Install Python dependencies
cd ../engine
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the `server` directory:

```env
PORT=3000
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Run

```bash
# Start the server
cd commander-arena/server
npm start
```

Open `http://localhost:3000` in your browser.

---

## 📁 Project Structure

```
mtg-ai-commander/
├── commander-arena/
│   ├── api/
│   │   └── main.py              # Python API entry point
│   ├── engine/
│   │   ├── actions.py           # Game actions and move validation
│   │   ├── card_parser.py       # Card data parsing
│   │   ├── effects.py           # Card effect resolution
│   │   ├── game_state.py        # Core game state management
│   │   ├── heuristic_ai.py      # Heuristic AI decision engine
│   │   ├── rules.py             # MTG rules enforcement
│   │   ├── simulator.py         # Game simulation logic
│   │   └── card_cache/          # Scryfall bulk data (gitignored)
│   ├── ml/
│   │   ├── model.py             # ML model for AI decision-making
│   │   └── train.py             # Model training scripts
│   └── server/
│       ├── server.js            # Express server
│       └── public/
│           └── index.html       # Frontend UI and board state
├── commander deck list.txt      # Physical deck list
└── README.md
```

---

## 🤖 How the AI Works

The project uses a three-phase AI approach:

**Phase 1: Claude API** — Anthropic Claude API handles AI decisions while the custom engine matures. Game state is serialized and sent to Claude, which returns structured actions the engine validates and executes.

**Phase 2: Heuristic Engine** — `heuristic_ai.py` provides a rule-based AI that evaluates board state using handcrafted MTG strategy heuristics. This runs locally with no API dependency.

**Phase 3: ML Model (In Development)** — A custom machine learning model (`ml/`) is being trained to replace both Claude and the heuristic engine, producing a fully self-contained AI trained specifically on Commander gameplay.

The AI turn pipeline:
1. Game state serialized and passed to the active AI backend
2. AI returns a structured action (play card, attack, pass, etc.)
3. Engine validates the action against MTG rules
4. Board state updates and renders to the UI
5. Abort timeout resets if the turn completes successfully

---

## 🃏 Scryfall Integration

Card data is fetched from the [Scryfall API](https://scryfall.com/docs/api) at runtime:
- Card names, mana costs, types, and oracle text
- Card artwork for the visual board
- Rulings for complex card interactions

Bulk card data is downloaded separately and stored in `card_cache/` (excluded from the repo via `.gitignore`). The app fetches individual cards on demand during gameplay.

---

## 🎙 Voice Commands

Supported voice commands (via Web Speech API):
- *"Play [card name]"* — Play a card from your hand
- *"Attack with [creature]"* — Declare attackers
- *"Pass turn"* — End your turn
- *"Show hand"* — Display your current hand

---

## 🐛 Known Limitations

- Currently supports 1v1 and basic multiplayer (full 4-player pod in progress)
- Voice recognition accuracy depends on browser and microphone quality
- Claude API dependency will be removed once the ML engine is complete

---

## 🗺 Roadmap

- [ ] Complete Python ML model to replace Claude API and heuristic engine
- [ ] Train model on Commander game data
- [ ] Full 4-player Commander pod support
- [ ] Deck builder UI for importing decklists
- [ ] Persistent game history and stats
- [ ] Mobile-responsive board layout

---

## 📄 License

MIT
