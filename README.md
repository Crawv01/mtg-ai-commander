# ⚔️ Commander AI Arena

A browser-based Magic: The Gathering Commander application where you play your physical deck against Claude-powered AI opponents. Features real-time card lookups via the Scryfall API, voice command support, and a visual board state — all running in the browser with a Node.js backend.

![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude_API-191919?style=flat&logo=anthropic&logoColor=white)

---

## ✨ Features

- **Claude-Powered AI Opponents** — Each AI player uses the Anthropic API to make strategic decisions based on the current board state, hand, and game context
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
| AI | Anthropic Claude API |
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
│   │   ├── ai_player.js         # Claude API integration for AI decisions
│   │   └── scryfall.js          # Scryfall API card lookups
│   ├── ml/
│   │   ├── model.py             # ML model for card evaluation
│   │   └── train.py             # Training scripts
│   └── server/
│       ├── server.js            # Express server
│       ├── public/
│       │   └── index.html       # Frontend UI and board state
│       └── .env                 # API keys (not committed)
└── README.md
```

---

## 🤖 How the AI Works

Each AI opponent is given a system prompt describing the Commander format rules, its current hand, the board state, life totals, and available mana. Claude then decides which cards to play, which creatures to attack with, and how to respond to the human player's actions.

The AI turn pipeline:
1. Game state serialized and sent to Claude API
2. Claude returns a structured action (play card, attack, pass, etc.)
3. Engine validates the action against game rules
4. Board state updates and renders to the UI
5. Abort timeout resets if turn completes successfully

---

## 🃏 Scryfall Integration

Card data is fetched from the [Scryfall API](https://scryfall.com/docs/api) at runtime:
- Card names, mana costs, types, and oracle text
- Card artwork for the visual board
- Rulings for complex card interactions

Bulk card data is downloaded separately and excluded from the repo (too large for GitHub). The app fetches individual cards on demand during gameplay.

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
- AI decision speed depends on Anthropic API response time

---

## 🗺 Roadmap

- [ ] Full 4-player Commander pod support
- [ ] Deck builder UI for importing decklists
- [ ] Persistent game history and stats
- [ ] Mobile-responsive board layout
- [ ] Additional AI difficulty levels

---

## 📄 License

MIT
