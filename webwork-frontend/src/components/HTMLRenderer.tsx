import React, { useEffect, useRef } from 'react';

interface HTMLRendererProps {
  html: string;
  className?: string;
}

/**
 * Renders HTML content with LaTeX math rendering
 * Processes both display math (delimited by \[ \]) and inline math (delimited by \( \))
 */
const HTMLRenderer: React.FC<HTMLRendererProps> = ({ html, className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Function to render math using KaTeX
    const renderMath = async () => {
      // Dynamically import katex to avoid SSR issues
      const katex = await import('katex');

      if (!containerRef.current) return;

      const container = containerRef.current;

      // Find all text nodes and process math delimiters
      const processNode = (node: Node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          const text = node.textContent || '';

          // Check for display math \[ ... \]
          const displayMathRegex = /\\\[(.*?)\\\]/gs;
          // Check for inline math \( ... \)
          const inlineMathRegex = /\\\((.*?)\\\)/gs;

          if (displayMathRegex.test(text) || inlineMathRegex.test(text)) {
            const span = document.createElement('span');
            let processedText = text;

            // Replace display math
            processedText = processedText.replace(displayMathRegex, (match, math) => {
              const mathSpan = document.createElement('span');
              mathSpan.className = 'math-display block my-4';
              try {
                katex.render(math.trim(), mathSpan, {
                  displayMode: true,
                  throwOnError: false,
                });
              } catch (e) {
                mathSpan.textContent = match;
              }
              return mathSpan.outerHTML;
            });

            // Replace inline math
            processedText = processedText.replace(inlineMathRegex, (match, math) => {
              const mathSpan = document.createElement('span');
              mathSpan.className = 'math-inline';
              try {
                katex.render(math.trim(), mathSpan, {
                  displayMode: false,
                  throwOnError: false,
                });
              } catch (e) {
                mathSpan.textContent = match;
              }
              return mathSpan.outerHTML;
            });

            span.innerHTML = processedText;
            node.parentNode?.replaceChild(span, node);
          }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
          // Don't process math inside script tags or already-rendered math
          const element = node as HTMLElement;
          if (
            element.tagName !== 'SCRIPT' &&
            !element.classList.contains('katex') &&
            !element.classList.contains('katex-html')
          ) {
            Array.from(node.childNodes).forEach(processNode);
          }
        }
      };

      processNode(container);
    };

    renderMath();
  }, [html]);

  return (
    <div
      ref={containerRef}
      className={`problem-content ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

export default HTMLRenderer;
