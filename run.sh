#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "====================================="
echo " களம் 🌾 — VIT Mapathon 2026"
echo "====================================="
echo ""

# Demo mode
if [ "$DEMO" = "1" ]; then
  echo "⚠️  DEMO MODE: serving from cache/outputs, no network calls"
fi

# Start backend
echo "🚀 Starting backend on http://localhost:8000 ..."
cd "$ROOT"
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Start frontend
echo "🌐 Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

# Trap Ctrl+C
trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "✅ Both services running."
echo "   Backend:  http://localhost:8000/docs"
echo "   Frontend: http://localhost:5173"
echo ""
wait
