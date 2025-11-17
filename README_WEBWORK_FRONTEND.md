# WebWork Python Frontend

A modern React/TypeScript frontend for WebWork Python problems with full mathematical rendering support.

## Features

- **React 18** with TypeScript for type safety
- **Tailwind CSS** for responsive, modern styling
- **KaTeX** for fast, beautiful mathematical rendering
- **Markdown support** with `react-markdown` for problem text
- **Real-time answer validation** and grading feedback
- **Multiple problem types**: formulas, numeric, vectors, matrices, etc.
- **LaTeX math rendering** in both inline `\(...\)` and display `\[...\]` modes
- **Responsive design** that works on desktop and mobile

## Prerequisites

- Node.js 18+ and npm (or yarn/pnpm)
- WebWork Python API running (see `webwork_api/` directory)

## Installation

```bash
cd webwork-frontend
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

The development server includes:
- Hot module replacement (HMR)
- Proxy to API at `http://localhost:8000`
- TypeScript type checking

## Building for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

Preview production build:

```bash
npm run preview
```

## Project Structure

```
webwork-frontend/
├── src/
│   ├── components/
│   │   ├── AnswerInput.tsx        # Answer input with feedback
│   │   ├── HTMLRenderer.tsx       # HTML + LaTeX rendering
│   │   ├── MarkdownRenderer.tsx   # Markdown rendering
│   │   ├── Problem.tsx            # Main problem component
│   │   └── ProblemSelector.tsx    # Problem picker
│   ├── api.ts                      # API client
│   ├── types.ts                    # TypeScript types
│   ├── App.tsx                     # Main app component
│   ├── main.tsx                    # Entry point
│   └── index.css                   # Global styles + Tailwind
├── index.html                      # HTML template
├── package.json                    # Dependencies
├── tsconfig.json                   # TypeScript config
├── tailwind.config.js              # Tailwind config
├── vite.config.ts                  # Vite config
└── postcss.config.js               # PostCSS config
```

## Key Technologies

### UI Framework
- **React 18**: Modern React with hooks
- **TypeScript**: Full type safety
- **Vite**: Fast development and building

### Styling
- **Tailwind CSS**: Utility-first CSS framework
- **Custom CSS**: For math rendering and feedback

### Math Rendering
- **KaTeX**: Fast math typesetting
- **remark-math** + **rehype-katex**: Markdown math integration

### API Communication
- **Axios**: HTTP client with TypeScript support
- **Proxy**: Development proxy to FastAPI backend

## Component Overview

### `Problem`
Main component that orchestrates:
- Loading problems from API
- Managing answer state
- Submitting answers for grading
- Displaying feedback and scores
- Showing hints and solutions

### `AnswerInput`
Reusable input component with:
- Support for different answer types
- Visual feedback (correct/incorrect/partial)
- Responsive sizing
- Disabled state during grading

### `HTMLRenderer`
Renders problem HTML with:
- KaTeX math rendering
- Support for `\[...\]` (display) and `\(...\)` (inline) LaTeX
- Automatic math detection and rendering

### `MarkdownRenderer`
Renders markdown content with:
- Full GFM (GitHub Flavored Markdown) support
- Math support via remark-math
- Code block syntax highlighting

### `ProblemSelector`
Dropdown selector for:
- Listing available problems
- Auto-loading problem list
- Formatting problem names

## API Integration

The frontend communicates with the FastAPI backend via:

```typescript
// Get a problem
GET /api/problems/{problemId}?seed={seed}

// Grade answers
POST /api/problems/{problemId}/grade
{
  "answers": { "AnSwEr0001": "2*x + 3" },
  "seed": 12345
}
```

API responses use Pydantic models for type safety.

## Customization

### Adding New Problem Types

1. Update `AnswerInput.tsx` to handle new input types
2. Add type-specific rendering in `HTMLRenderer.tsx`
3. Update TypeScript types in `types.ts`

### Styling

Modify `tailwind.config.js` for:
- Custom colors
- Fonts
- Spacing
- Breakpoints

Edit `src/index.css` for:
- Global styles
- Math rendering styles
- Feedback colors

### Math Rendering

Configure KaTeX options in `HTMLRenderer.tsx`:

```typescript
katex.render(math, element, {
  displayMode: true,
  throwOnError: false,
  // Add more options...
});
```

## Troubleshooting

### Math not rendering
- Check that KaTeX CSS is imported in `index.css`
- Verify LaTeX delimiters: `\[...\]` for display, `\(...\)` for inline
- Check browser console for KaTeX errors

### API connection issues
- Ensure FastAPI backend is running on port 8000
- Check Vite proxy configuration in `vite.config.ts`
- Verify CORS settings in FastAPI

### TypeScript errors
- Run `npm run build` to check for type errors
- Update types in `types.ts` to match API responses
- Check `tsconfig.json` settings

## Performance

- **Code splitting**: Vite automatically splits code
- **Lazy loading**: Components load on demand
- **KaTeX caching**: Math rendering is cached
- **React optimization**: Uses React.memo and hooks efficiently

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

Same as WebWork Python (check main repository)
