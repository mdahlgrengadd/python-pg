#!/bin/bash
# Startup script for WebWork Python frontend + backend

echo "==================================================="
echo "WebWork Python - Startup Script"
echo "==================================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python && ! command_exists python3; then
    echo "❌ Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

echo "✓ Python found: $(python3 --version 2>/dev/null || python --version)"
echo "✓ Node.js found: $(node --version)"
echo "✓ npm found: $(npm --version)"
echo ""

# Determine Python command
if command_exists python3; then
    PYTHON=python3
else
    PYTHON=python
fi

# Check if dependencies are installed
echo "Checking dependencies..."

# Check Python dependencies
if ! $PYTHON -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not found. Installing backend dependencies..."
    cd webwork_api
    pip install -r requirements.txt
    cd ..
fi

# Check if pg package is installed
if ! $PYTHON -c "import pg" 2>/dev/null; then
    echo "⚠️  pg package not found. Installing..."
    pip install -e .
fi

# Check Node dependencies
if [ ! -d "webwork-frontend/node_modules" ]; then
    echo "⚠️  Node modules not found. Installing frontend dependencies..."
    cd webwork-frontend
    npm install
    cd ..
fi

echo "✓ All dependencies installed"
echo ""

# Start backend
echo "Starting FastAPI backend on http://localhost:8000..."
cd webwork_api
$PYTHON main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ Failed to start backend"
    exit 1
fi

echo "✓ Backend started (PID: $BACKEND_PID)"
echo ""

# Start frontend
echo "Starting React frontend on http://localhost:3000..."
cd webwork-frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

# Check if frontend started successfully
if ! ps -p $FRONTEND_PID > /dev/null; then
    echo "❌ Failed to start frontend"
    kill $BACKEND_PID
    exit 1
fi

echo "✓ Frontend started (PID: $FRONTEND_PID)"
echo ""
echo "==================================================="
echo "✓ WebWork Python is now running!"
echo "==================================================="
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ Servers stopped"
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT TERM

# Wait for user interrupt
wait
