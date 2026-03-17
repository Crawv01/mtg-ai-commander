import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { createServer } from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3001;

// ─── Validate environment ────────────────────────────────────────────────────
if (!process.env.ANTHROPIC_API_KEY) {
  console.error('❌  ANTHROPIC_API_KEY is not set. Create a .env file. Exiting.');
  process.exit(1);
}

// ─── Security middleware ─────────────────────────────────────────────────────
app.use(helmet({
  contentSecurityPolicy: false, // We'll tighten this later when serving the client
}));

// CORS — in dev allow all origins, in prod lock to your domain
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',').map(o => o.trim())
  : ['http://localhost:3000', 'http://localhost:3001',
     'http://localhost:3002', 'http://localhost:5173'];

app.use(cors({
  origin: (origin, cb) => {
    // Allow requests with no origin (curl, Postman, same-origin)
    if (!origin || allowedOrigins.includes(origin)) return cb(null, true);
    // Also allow any localhost origin in development
    if (origin && origin.startsWith('http://localhost')) return cb(null, true);
    cb(new Error(`CORS: origin ${origin} not allowed`));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));

app.use(express.json({ limit: '50kb' })); // Prevent giant payloads

// ─── Rate limiting ───────────────────────────────────────────────────────────
// Global limiter — 500 requests per 15 minutes per IP
// (a full 4-player game can burn ~335 requests)
const globalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 500,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please slow down.' },
});

// Claude endpoint limiter — ~3 full games per hour per IP
// A full game uses ~335 Claude calls. 3 games = ~1000 calls/hr = ~17/min.
// We use per-hour window so bursty gameplay (many calls in a short turn) works fine.
// NOTE: This protects against abuse, not cost. See README for cost management strategy.
const claudeLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,   // 1 hour window
  max: 1000,                   // ~3 full games per hour per IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Hourly game limit reached. Come back in a bit, or use your own API key.' },
});

app.use(globalLimiter);

// ─── Allowed Claude models ───────────────────────────────────────────────────
// Whitelist prevents client from requesting expensive models
const ALLOWED_MODELS = new Set([
  'claude-haiku-4-5-20251001',
  'claude-sonnet-4-20250514',
  'claude-sonnet-4-6',
]);

const MAX_TOKENS_LIMIT = 2000; // Hard cap — client can request less, never more

// ─── /api/claude proxy ───────────────────────────────────────────────────────
app.post('/api/claude', claudeLimiter, async (req, res) => {
  const { model, max_tokens, messages, system } = req.body;

  // ── Validation ──────────────────────────────────────────────────
  if (!model || !messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'Invalid request: model and messages are required.' });
  }

  if (!ALLOWED_MODELS.has(model)) {
    return res.status(400).json({ error: `Model '${model}' is not permitted.` });
  }

  if (!Number.isInteger(max_tokens) || max_tokens < 1 || max_tokens > MAX_TOKENS_LIMIT) {
    return res.status(400).json({ error: `max_tokens must be between 1 and ${MAX_TOKENS_LIMIT}.` });
  }

  // Validate message structure — prevent prompt injection via malformed messages
  for (const msg of messages) {
    if (!['user', 'assistant'].includes(msg.role)) {
      return res.status(400).json({ error: 'Invalid message role.' });
    }
    if (typeof msg.content !== 'string' || msg.content.length > 20000) {
      return res.status(400).json({ error: 'Message content must be a string under 20,000 chars.' });
    }
  }

  // ── Proxy to Anthropic ──────────────────────────────────────────
  try {
    const body = { model, max_tokens, messages };
    if (system && typeof system === 'string') body.system = system;

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000), // 30s hard timeout
    });

    const data = await response.json();

    if (!response.ok) {
      console.error(`[Claude API] ${response.status}:`, data);
      return res.status(response.status).json({
        error: data.error?.message || 'Anthropic API error',
      });
    }

    return res.json(data);

  } catch (err) {
    if (err.name === 'TimeoutError') {
      return res.status(504).json({ error: 'Claude API timed out.' });
    }
    console.error('[/api/claude] Unexpected error:', err);
    return res.status(500).json({ error: 'Internal server error.' });
  }
});


// ─── Python engine proxy ─────────────────────────────────────────────────────
// Forwards /api/engine/* requests to the FastAPI Python engine on port 3002.
// Falls back gracefully if the engine isn't running.
const ENGINE_URL = process.env.ENGINE_URL || 'http://localhost:3002';

app.all('/api/engine/*', async (req, res) => {
  const targetUrl = ENGINE_URL + req.path + (req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '');
  try {
    const engineRes = await fetch(targetUrl, {
      method:  req.method,
      headers: { 'Content-Type': 'application/json', 'Origin': 'http://localhost:3001' },
      body:    req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
      signal:  AbortSignal.timeout(30000),
    });
    const engineData = await engineRes.json();
    return res.status(engineRes.status).json(engineData);
  } catch (err) {
    if (err.name === 'TimeoutError') {
      return res.status(504).json({ error: 'Python engine timed out.' });
    }
    // Engine not running — return a clear error
    return res.status(503).json({
      error: 'Python engine unavailable. Start it with: cd api && uvicorn main:app --port 3002',
      detail: err.message,
    });
  }
});


// ─── Serve local card database ───────────────────────────────────────────────
// Serves the cards.json downloaded by fetch_precons.py
// Frontend uses this instead of Scryfall API for instant, free card lookups
import { existsSync } from 'fs';
const CARDS_JSON_PATH = path.join(__dirname, '..', 'engine', 'card_cache', 'cards.json');

app.get('/cards.json', (req, res) => {
  if (existsSync(CARDS_JSON_PATH)) {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'public, max-age=86400'); // Cache for 1 day
    res.sendFile(CARDS_JSON_PATH);
  } else {
    res.status(404).json({ error: 'cards.json not found. Run fetch_precons.py first.' });
  }
});

// ─── Serve the game client ───────────────────────────────────────────────────
// Serves index.html from the /public folder so everything runs on one port
app.use(express.static(path.join(__dirname, 'public')));

// Catch-all so refreshing the page works (for future React router)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ─── Health check ────────────────────────────────────────────────────────────
app.get('/health', (req, res) => res.json({ status: 'ok', ts: Date.now() }));

// ─── Start ───────────────────────────────────────────────────────────────────
const server = createServer(app);
server.listen(PORT, () => {
  console.log(`\n✅  Commander Arena server running`);
  console.log(`   Local:  http://localhost:${PORT}`);
  console.log(`   Env:    ${process.env.NODE_ENV || 'development'}\n`);
});

// Graceful shutdown
process.on('SIGTERM', () => { server.close(() => process.exit(0)); });
process.on('SIGINT',  () => { server.close(() => process.exit(0)); });
