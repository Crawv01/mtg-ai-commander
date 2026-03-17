# Commander AI Arena

A browser-based Magic: The Gathering Commander game where you play your physical deck against Claude-powered AI opponents.

---

## Project Structure

```
commander-arena/
├── server/
│   ├── server.js          ← Express proxy server (keeps API key secure)
│   ├── public/            ← Drop index.html here to serve the game
│   │   └── index.html
│   ├── .env               ← Your secrets (never committed to git)
│   ├── .env.example       ← Template — copy this to .env
│   ├── .gitignore
│   └── package.json
└── README.md
```

---

## First-Time Setup

### 1. Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

### 2. Install dependencies

```bash
cd server
npm install
```

### 3. Configure environment

```bash
# In the server/ directory
cp .env.example .env
```

Open `.env` and replace the placeholder with your real Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

### 4. Add the game file

Copy your `index.html` into `server/public/`:

```bash
cp /path/to/index.html server/public/index.html
```

### 5. Start the server

```bash
# In the server/ directory
npm run dev      # Development (auto-restarts on file changes)
# or
npm start        # Production
```

Open your browser to **http://localhost:3001**

---

## How the Proxy Works

The game client (`index.html`) makes requests to `/api/claude` — a relative URL. The Express server:

1. Receives the request
2. Validates the model, token limit, and message structure
3. Adds your API key from the `.env` file (never exposed to the browser)
4. Forwards the request to Anthropic
5. Returns the response to the client

Your API key **never leaves the server**.

---

## Security Features

| Feature | Details |
|---|---|
| API key isolation | Key stored in `.env`, never sent to client |
| Model whitelist | Client can only use approved Claude models |
| Token cap | Hard limit of 2,000 tokens per request |
| Rate limiting | 30 Claude calls/min per IP, 200 requests/15min globally |
| Payload size limit | Request body capped at 50KB |
| CORS | Configurable origin allowlist |
| Helmet | Standard HTTP security headers |

---

## Deployment (Render.com — Recommended)

Render is the easiest free option and what your real estate app already uses.

1. Push this repo to GitHub
2. Create a new **Web Service** on Render
3. Set **Build Command**: `cd server && npm install`
4. Set **Start Command**: `cd server && npm start`
5. Add environment variables in the Render dashboard:
   - `ANTHROPIC_API_KEY` = your key
   - `NODE_ENV` = production
   - `ALLOWED_ORIGINS` = https://your-app.onrender.com
6. Deploy

---

## Development Workflow

```
server/public/index.html   ← Edit your game here
server/server.js           ← Edit proxy config here

# When you change index.html:
# Just refresh the browser — no rebuild needed

# When you change server.js:
# nodemon auto-restarts (npm run dev), or manually restart (npm start)
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | — | Your Anthropic API key |
| `PORT` | No | 3001 | Port the server listens on |
| `NODE_ENV` | No | development | Environment mode |
| `ALLOWED_ORIGINS` | No | localhost:3000, localhost:5173 | CORS origin allowlist |
