# WebWork React Frontend - Complete Setup Guide

This guide will help you set up and run the complete WebWork Python system with a React frontend.

## Overview

The system consists of two parts:

1. **FastAPI Backend** (`webwork_api/`) - Serves problems and grades answers
2. **React Frontend** (`webwork-frontend/`) - User interface for solving problems

## Prerequisites

### Backend Requirements
- Python 3.10 or higher
- pip (Python package manager)

### Frontend Requirements
- Node.js 18 or higher
- npm (comes with Node.js)

### Check Your Versions

```bash
# Check Python version
python --version  # or python3 --version

# Check Node.js version
node --version

# Check npm version
npm --version
```

## Quick Start

### 1. Install Python Dependencies

From the project root directory:

```bash
# Install the WebWork Python package
pip install -e .

# Install FastAPI dependencies
cd webwork_api
pip install -r requirements.txt
cd ..
```

### 2. Start the Backend (FastAPI)

In one terminal:

```bash
cd webwork_api
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

The API is now running at `http://localhost:8000`

**Test it**: Open `http://localhost:8000/docs` in your browser to see the interactive API documentation.

### 3. Install Frontend Dependencies

In a **new terminal** (keep the backend running):

```bash
cd webwork-frontend
npm install
```

This will install all required packages (React, TypeScript, Tailwind, KaTeX, etc.).

### 4. Start the Frontend (React)

```bash
npm run dev
```

You should see:
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

The frontend is now running at `http://localhost:3000`

### 5. Open in Browser

Open `http://localhost:3000` in your web browser.

You should see:
- A "WebWork Python" header
- A problem selector dropdown
- A random seed input
- A problem display area

## Usage

### Solving a Problem

1. **Select a problem** from the dropdown (e.g., "Simple Algebra")
2. The problem will load and display with answer boxes
3. **Enter your answer** in the input field
4. Click **"Submit Answers"** to grade your work
5. **View feedback**:
   - Green = Correct
   - Red = Incorrect
   - Yellow = Partial credit

### Features to Try

- **Change the seed**: Enter a different number or click "🎲 New" to get a different version of the problem
- **View hints**: Click the "▶ Hint" button if available
- **View solutions**: Click the "▶ Solution" button to see the full solution
- **Try again**: After submitting, click "Try Again" to reset

## Available Problems

The system comes with three example problems:

1. **simple_algebra** - Expand an algebraic expression
2. **quadratic** - Solve a linear equation
3. **calculus_derivative** - Find a derivative using the power rule

### Adding Your Own Problems

Create a `.pg` file in `webwork_api/problems/`:

```perl
DOCUMENT();
loadMacros("PGstandard.pl", "PGML.pl", "MathObjects.pl");
TEXT(beginproblem());

Context("Numeric");
$a = random(1, 10, 1);
$b = random(1, 10, 1);
$answer = Real($a + $b);

BEGIN_PGML
What is [$a] + [$b]?

[_]{$answer}
END_PGML

ENDDOCUMENT();
```

Save as `webwork_api/problems/my_problem.pg` and it will appear in the dropdown!

## Project Structure

```
python-pg/
├── pg/                          # WebWork Python core library
├── webwork_api/                 # FastAPI backend
│   ├── main.py                  # API server
│   ├── requirements.txt         # Python dependencies
│   └── problems/                # Problem files (.pg)
│       ├── simple_algebra.pg
│       ├── quadratic.pg
│       └── calculus_derivative.pg
└── webwork-frontend/            # React frontend
    ├── src/
    │   ├── components/          # React components
    │   ├── api.ts              # API client
    │   ├── types.ts            # TypeScript types
    │   ├── App.tsx             # Main app
    │   └── main.tsx            # Entry point
    ├── package.json            # npm dependencies
    └── vite.config.ts          # Build configuration
```

## Troubleshooting

### Backend Issues

**Problem**: "Module 'pg' not found"
```bash
# Solution: Install the pg package
pip install -e .  # From python-pg root directory
```

**Problem**: Port 8000 already in use
```bash
# Solution: Change the port in main.py or kill the process
# Option 1: Change port
uvicorn main:app --port 8001

# Option 2: Kill process (Linux/Mac)
lsof -ti:8000 | xargs kill -9
```

**Problem**: CORS errors in browser console
```bash
# Solution: Check that CORS is enabled in main.py
# (It should be by default)
```

### Frontend Issues

**Problem**: "Cannot connect to API"
```bash
# Solution: Make sure backend is running on port 8000
# Check: http://localhost:8000/docs
```

**Problem**: Math not rendering
```bash
# Solution: Make sure KaTeX CSS is imported
# Check src/index.css has: @import 'katex/dist/katex.min.css';
```

**Problem**: Port 3000 already in use
```bash
# Solution: Vite will automatically use next available port
# Or specify a different port in vite.config.ts
```

**Problem**: `npm install` fails
```bash
# Solution: Clear npm cache and try again
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## Development Tips

### Hot Reload

Both the backend and frontend support hot reload:

- **Backend**: Automatically reloads when you edit Python files
- **Frontend**: Automatically reloads when you edit React files

### API Documentation

The FastAPI backend provides interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Browser DevTools

- **React DevTools**: Install the React DevTools browser extension
- **Console**: Open browser console (F12) to see logs and errors
- **Network Tab**: View API requests and responses

### TypeScript

The frontend uses TypeScript for type safety:

```bash
# Check for type errors
cd webwork-frontend
npm run build
```

## Production Deployment

### Backend

```bash
# Install production dependencies
pip install uvicorn[standard] gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Frontend

```bash
# Build for production
cd webwork-frontend
npm run build

# Serve static files (dist/ directory)
# Use nginx, Apache, or any static file server
```

## Next Steps

1. **Explore the example problems** to understand the format
2. **Create your own problems** in the `problems/` directory
3. **Customize the frontend** styling in Tailwind config
4. **Add authentication** for student tracking
5. **Implement a database** for saving student progress
6. **Deploy to production** using Docker or cloud platforms

## Getting Help

- **Backend Issues**: Check `webwork_api/README.md`
- **Frontend Issues**: Check `README_WEBWORK_FRONTEND.md`
- **PG Format**: Refer to WebWork documentation
- **FastAPI**: https://fastapi.tiangolo.com
- **React**: https://react.dev
- **Tailwind**: https://tailwindcss.com

## License

WebWork Python is open source. See the main repository for license details.
