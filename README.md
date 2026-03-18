# ⚔️ Commander AI Arena

A browser-based Magic: The Gathering Commander application where you play your physical deck against AI opponents. Features real-time card lookups via the Scryfall API, voice command support, and a visual board state — all running in the browser with a Node.js backend. Currently powered by the Anthropic Claude API with a custom Python ML engine in development as a full replacement.

![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude_API-191919?style=flat&logo=anthropic&logoColor=white)

---

## ✨ Features

- **AI Opponents** — AI players make strategic decisions based on board state, hand, and game context — currently via Claude API, with a custom Python ML engine in development as a replacement
- **Scryfall API Integration** — Real-time card data, rulings, and artwork fetched directly from Scryfall's database
- **Voice Commands** — Web Speech API lets you speak actions instead of clicking
- **Visual Board State** — Interactive UI displays each player's board, hand, life total, and graveyard
- **Commander Format Rules** — Handles Commander damage, command zone, color identity, and multiplayer logic
- **Debug Panel** — Built-in debug overlay for monitoring AI turn state and diagnosing stalls
- **Abort Timeouts** — Prevents AI turns from hanging indefinitely with graceful error recovery

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | Node.js, Express |
| AI (Current) | Anthropic Claude API |
| AI (In Development) | Custom Python ML Engine |
| Card Data | Scryfall API |
| Voice | Web Speech API |

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Anthropic API key

### Installation

```bash
git clone https://github.com/Crawv01/mtg-ai-commander.git
cd mtg-ai-commander/server
npm install
```

### Environment Variables

Create a `.env` file in the `server` directory:

```env
PORT=3000
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Run

```bash
cd server
npm start
```

Open `http://localhost:3000` in your browser.

---

## 📁 Project Structure

```
mtg-ai-commander/
├── commander-arena/
│   ├── engine/
│   │   ├── game_engine.js       # Core game state and turn logic
│   │   ├── ai_player.js         # AI decision interface (Claude or ML engine)
│   │   └── scryfall.js          # Scryfall API card lookups
│   ├── ml/
│   │   ├── model.py             # Custom ML model for AI decision-making
│   │   └── train.py             # Model training scripts
│   └── server/
│       ├── server.js            # Express server
│       ├── public/
│       │   └── index.html       # Frontend UI and board state
│       └── .env                 # API keys (not committed)
└── README.md
```

---

## 🤖 How the AI Works

The project uses a two-phase AI approach:

**Current: Claude API** — AI opponents currently use the Anthropic Claude API to make strategic decisions based on the serialized board state, hand, life totals, and available mana. Claude returns structured actions (play card, attack, pass, etc.) which the engine validates and executes.

**In Development: Custom Python ML Engine** — A custom machine learning engine (`commander-arena/ml/`) is being built to replace Claude entirely. The goal is a self-contained AI trained specifically on MTG Commander decision-making, eliminating the dependency on an external API.

The AI turn pipeline:
1. Game state serialized and passed to the active AI backend
2. AI returns a structured action (play card, attack, pass, etc.)
3. Engine validates the action against game rules
4. Board state updates and renders to the UI
5. Abort timeout resets if turn completes successfully

---

## 🃏 Scryfall Integration

Card data is fetched from the [Scryfall API](https://scryfall.com/docs/api) at runtime:
- Card names, mana costs, types, and oracle text
- Card artwork for the visual board
- Rulings for complex card interactions

Bulk card data is downloaded separately and excluded from the repo. The app fetches individual cards on demand during gameplay.

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
- AI decision speed depends on Anthropic API response time (will be resolved once ML engine is complete)

---

## 🗺 Roadmap

- [ ] Complete Python ML engine to replace Claude API dependency
- [ ] Train model on Commander game data
- [ ] Full 4-player Commander pod support
- [ ] Deck builder UI for importing decklists
- [ ] Persistent game history and stats
- [ ] Mobile-responsive board layout
- [ ] Additional AI difficulty levels

---

## 📄 License

MIT
