/// <reference types="vite/client" />

interface PuterAI {
  chat(prompt: string, options?: { model?: string }): Promise<string>
}

interface PuterGlobal {
  ai?: PuterAI
}

interface Window {
  puter?: PuterGlobal
}
