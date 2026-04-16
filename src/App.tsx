import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { createClient } from '@supabase/supabase-js'
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

type ChatRole = 'user' | 'assistant'
type RoutingMode = 'auto' | 'manual' | 'agent'
type ModelName = 'claude' | 'gpt' | 'gemini' | 'groq' | 'image' | 'video'

type ChatMessage = {
  role: ChatRole
  content: string
  model?: ModelName
}

let conversationHistory: Array<{ role: ChatRole; content: string }> = []
let lastModelUsed: ModelName | null = null

GlobalWorkerOptions.workerSrc = pdfWorker

const supabase =
  import.meta.env.VITE_SUPABASE_URL && import.meta.env.VITE_SUPABASE_ANON_KEY
    ? createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY)
    : null

function detectModel(input: string): ModelName {
  const text = input.toLowerCase()

  if (/(video|animate)/.test(text)) return 'video'
  if (/(image|draw|photo)/.test(text)) return 'image'
  if (/(code|debug|error)/.test(text)) return 'claude'
  if (/(write|essay|story|blog)/.test(text)) return 'gemini'
  if (/(news|latest|today)/.test(text)) return 'groq'

  return 'gpt'
}

const keywords = {
  coding: ['code', 'debug', 'bug', 'stack trace', 'typescript', 'react', 'function'],
  writing: ['write', 'essay', 'story', 'blog', 'poem', 'draft', 'content'],
  realtime: ['news', 'latest', 'today', 'trend', 'current'],
  general: ['explain', 'plan', 'summarize', 'help'],
}

function score(prompt: string, terms: string[]): number {
  const text = prompt.toLowerCase()
  return terms.reduce((sum, term) => sum + (text.includes(term) ? 1 : 0), 0)
}

function highestScore(scores: Record<Exclude<ModelName, 'image' | 'video'>, number>): Exclude<ModelName, 'image' | 'video'> {
  return Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0] as Exclude<ModelName, 'image' | 'video'>
}

function agentSelect(prompt: string): Exclude<ModelName, 'image' | 'video'> {
  return highestScore({
    claude: score(prompt, keywords.coding),
    gemini: score(prompt, keywords.writing),
    groq: score(prompt, keywords.realtime),
    gpt: score(prompt, keywords.general),
  })
}

async function callGemini(prompt: string): Promise<string> {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${import.meta.env.VITE_GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
      }),
    },
  )

  if (!res.ok) {
    throw new Error(`Gemini failed: ${res.status}`)
  }

  const data = await res.json()
  return data?.candidates?.[0]?.content?.parts?.[0]?.text || 'No response'
}

async function callGroq(prompt: string): Promise<string> {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${import.meta.env.VITE_GROQ_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama3-70b-8192',
      messages: [{ role: 'user', content: prompt }],
    }),
  })

  if (!res.ok) {
    throw new Error(`Groq failed: ${res.status}`)
  }

  const data = await res.json()
  return data?.choices?.[0]?.message?.content || 'No response'
}

function generateImage(prompt: string): string {
  return `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`
}

async function replicateFallback(prompt: string): Promise<string> {
  const res = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${import.meta.env.VITE_REPLICATE_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ input: { prompt } }),
  })

  if (!res.ok) {
    throw new Error(`Replicate failed: ${res.status}`)
  }

  return 'Video processing...'
}

async function generateVideo(prompt: string): Promise<string> {
  try {
    if (!prompt.trim()) {
      throw new Error('Empty prompt')
    }
    return `https://gen.pollinations.ai/video/${encodeURIComponent(prompt)}`
  } catch {
    return replicateFallback(prompt)
  }
}

async function fallbackLLM(prompt: string): Promise<string> {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${import.meta.env.VITE_OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'openai/gpt-3.5-turbo',
      messages: [{ role: 'user', content: prompt }],
    }),
  })

  if (!res.ok) {
    throw new Error(`OpenRouter failed: ${res.status}`)
  }

  const data = await res.json()
  return data?.choices?.[0]?.message?.content || 'Fallback response'
}

async function callModel(model: ModelName, prompt: string): Promise<string> {
  console.log('🔥 Calling:', model)

  try {
    switch (model) {
      case 'claude':
        if (!window.puter?.ai) throw new Error('Puter unavailable')
        return await window.puter.ai.chat(prompt, { model: 'claude-sonnet-4' })
      case 'gpt':
        if (!window.puter?.ai) throw new Error('Puter unavailable')
        return await window.puter.ai.chat(prompt, { model: 'gpt-4o' })
      case 'gemini':
        return await callGemini(prompt)
      case 'groq':
        return await callGroq(prompt)
      case 'image':
        return generateImage(prompt)
      case 'video':
        return await generateVideo(prompt)
      default:
        if (!window.puter?.ai) throw new Error('Puter unavailable')
        return await window.puter.ai.chat(prompt)
    }
  } catch (e) {
    console.log('⚠️ fallback triggered', e)
    return await fallbackLLM(prompt)
  }
}

function badgeForModel(model: ModelName): string {
  switch (model) {
    case 'claude':
      return '⚡ Claude — Coding'
    case 'gemini':
      return '⚡ Gemini — Writing'
    case 'groq':
      return '⚡ Groq — Realtime'
    case 'image':
      return '🎨 Pollinations — Image'
    case 'video':
      return '🎬 Pollinations/Replicate — Video'
    default:
      return '⚡ GPT — General'
  }
}

async function extractFileText(file: File): Promise<string> {
  if (file.type === 'text/plain' || file.name.toLowerCase().endsWith('.txt')) {
    return file.text()
  }

  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    const data = new Uint8Array(await file.arrayBuffer())
    const pdf = await getDocument({ data }).promise
    let content = ''

    for (let pageNo = 1; pageNo <= pdf.numPages; pageNo += 1) {
      const page = await pdf.getPage(pageNo)
      const text = await page.getTextContent()
      const pageContent = text.items
        .map((item) => ('str' in item ? item.str : ''))
        .join(' ')
      content += `${pageContent}\n`
    }

    return content
  }

  throw new Error('Unsupported file type. Please upload txt or pdf.')
}

async function transcribeWithDeepgram(audioBlob: Blob): Promise<string> {
  if (!import.meta.env.VITE_DEEPGRAM_API_KEY) {
    throw new Error('Missing Deepgram API key')
  }

  const res = await fetch('https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true', {
    method: 'POST',
    headers: {
      Authorization: `Token ${import.meta.env.VITE_DEEPGRAM_API_KEY}`,
      'Content-Type': audioBlob.type || 'audio/webm',
    },
    body: audioBlob,
  })

  if (!res.ok) {
    throw new Error(`Deepgram failed: ${res.status}`)
  }

  const data = await res.json()
  return data?.results?.channels?.[0]?.alternatives?.[0]?.transcript || ''
}

function App() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<RoutingMode>('auto')
  const [manualModel, setManualModel] = useState<ModelName>('gpt')
  const [isRecording, setIsRecording] = useState(false)
  const [fileText, setFileText] = useState('')
  const [fileName, setFileName] = useState('')
  const [activeModel, setActiveModel] = useState<ModelName>('gpt')

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const endRef = useRef<HTMLDivElement | null>(null)

  const canSubmit = useMemo(() => prompt.trim().length > 0 && !isLoading, [prompt, isLoading])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    const loadHistory = async () => {
      if (!supabase) return
      const { data, error: loadError } = await supabase
        .from('history')
        .select('prompt,response,model,created_at')
        .order('created_at', { ascending: true })

      if (loadError || !data) return

      const hydrated: ChatMessage[] = []
      data.forEach((row) => {
        hydrated.push({ role: 'user', content: row.prompt })
        hydrated.push({ role: 'assistant', content: row.response, model: (row.model || 'gpt') as ModelName })
      })

      setMessages(hydrated)
      conversationHistory = hydrated.map((m) => ({ role: m.role, content: m.content }))
    }

    loadHistory().catch(() => undefined)
  }, [])

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const text = await extractFileText(file)
      setFileText(text.trim())
      setFileName(file.name)
      setError(null)
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : 'Failed to parse file.'
      setError(message)
    }
  }

  const handleMic = async () => {
    if (isRecording && recorderRef.current) {
      recorderRef.current.stop()
      setIsRecording(false)
      return
    }

    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []

      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        streamRef.current?.getTracks().forEach((track) => track.stop())
        streamRef.current = null

        try {
          const transcript = await transcribeWithDeepgram(blob)
          if (transcript) {
            setPrompt((previous) => (previous ? `${previous} ${transcript}` : transcript))
          }
        } catch (voiceError) {
          const message = voiceError instanceof Error ? voiceError.message : 'Voice transcription failed.'
          setError(message)
        }
      }

      recorder.start()
      setIsRecording(true)
    } catch (micError) {
      const message = micError instanceof Error ? micError.message : 'Unable to access microphone.'
      setError(message)
      setIsRecording(false)
    }
  }

  const submitPrompt = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmedPrompt = prompt.trim()
    if (!trimmedPrompt) return

    const fullPrompt = fileText
      ? `${trimmedPrompt}\n\n[Attached file: ${fileName}]\n${fileText}`
      : trimmedPrompt

    const selectedModel: ModelName =
      mode === 'manual'
        ? manualModel
        : mode === 'agent'
          ? agentSelect(fullPrompt)
          : detectModel(fullPrompt)

    setActiveModel(selectedModel)
    setPrompt('')
    setFileText('')
    setFileName('')
    setIsLoading(true)
    setError(null)

    const userMessage: ChatMessage = { role: 'user', content: trimmedPrompt }
    setMessages((previous) => [...previous, userMessage])

    const contextPrompt = `Conversation:\n${conversationHistory.map((m) => `${m.role}: ${m.content}`).join('\n')}\n\nUser: ${fullPrompt}\n`

    try {
      const response = await callModel(selectedModel, contextPrompt)
      const assistantMessage: ChatMessage = { role: 'assistant', content: response, model: selectedModel }

      setMessages((previous) => [...previous, assistantMessage])
      conversationHistory = [
        ...conversationHistory,
        { role: 'user', content: trimmedPrompt },
        { role: 'assistant', content: response },
      ]
      lastModelUsed = selectedModel

      if (supabase) {
        await supabase.from('history').insert({
          prompt: trimmedPrompt,
          response,
          model: selectedModel,
        })
      }
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : 'Something went wrong.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto flex h-screen max-w-5xl flex-col px-4 py-5">
        <header className="mb-4 rounded-2xl border border-slate-800 bg-slate-900/80 p-4 backdrop-blur">
          <h1 className="text-2xl font-bold tracking-tight">🚀 AI STATION</h1>
          <p className="text-sm text-slate-300">Every Query. The Right Intelligence.</p>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full bg-slate-800 px-3 py-1">Mode</span>
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as RoutingMode)}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1"
            >
              <option value="auto">Auto</option>
              <option value="manual">Manual</option>
              <option value="agent">Agent Mode</option>
            </select>
            {mode === 'manual' && (
              <select
                value={manualModel}
                onChange={(event) => setManualModel(event.target.value as ModelName)}
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1"
              >
                <option value="claude">Claude</option>
                <option value="gpt">GPT</option>
                <option value="gemini">Gemini</option>
                <option value="groq">Groq</option>
                <option value="image">Image</option>
                <option value="video">Video</option>
              </select>
            )}
            <span className="rounded-full bg-indigo-500/30 px-3 py-1">{badgeForModel(activeModel)}</span>
            {lastModelUsed && (
              <span className="rounded-full bg-emerald-500/20 px-3 py-1">Last: {lastModelUsed}</span>
            )}
          </div>
        </header>

        <section className="flex-1 space-y-3 overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          {messages.length === 0 ? (
            <p className="text-sm text-slate-400">Start by asking a question, uploading a file, or using voice input.</p>
          ) : (
            messages.map((message, index) => (
              <motion.article
                key={`${message.role}-${index}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={`rounded-xl p-3 text-sm ${
                  message.role === 'user' ? 'ml-8 bg-indigo-500/20' : 'mr-8 bg-slate-800'
                }`}
              >
                <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
                  <span>{message.role === 'user' ? 'You' : 'AI'}</span>
                  {message.model && <span>{badgeForModel(message.model)}</span>}
                </div>
                {message.model === 'image' && message.role === 'assistant' ? (
                  <img src={message.content} alt="Generated" className="max-h-80 rounded-lg" />
                ) : message.model === 'video' && message.role === 'assistant' ? (
                  <a
                    className="text-indigo-300 underline"
                    href={message.content}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open generated video
                  </a>
                ) : (
                  <p className="whitespace-pre-wrap">{message.content}</p>
                )}
              </motion.article>
            ))
          )}

          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mr-8 inline-flex items-center gap-2 rounded-xl bg-slate-800 p-3"
              >
                <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.3s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400 [animation-delay:-0.15s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" />
              </motion.div>
            )}
          </AnimatePresence>

          <div ref={endRef} />
        </section>

        {error && <p className="mt-2 text-sm text-rose-300">{error}</p>}

        {fileName && (
          <p className="mt-2 text-xs text-emerald-300">Attached: {fileName} (included in next prompt)</p>
        )}

        <form onSubmit={submitPrompt} className="mt-4 space-y-2">
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Ask anything…"
            className="h-28 w-full rounded-xl border border-slate-700 bg-slate-900 p-3 text-sm outline-none ring-indigo-500/40 focus:ring"
          />

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="submit"
              disabled={!canSubmit}
              className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-60"
            >
              Send
            </button>

            <button
              type="button"
              onClick={handleMic}
              className={`rounded-lg px-4 py-2 text-sm ${isRecording ? 'bg-rose-500' : 'bg-slate-700'}`}
            >
              {isRecording ? 'Stop Mic' : '🎤 Voice'}
            </button>

            <label className="cursor-pointer rounded-lg bg-slate-700 px-4 py-2 text-sm">
              📎 Upload txt/pdf
              <input
                type="file"
                accept=".txt,.pdf,text/plain,application/pdf"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </form>
      </main>
    </div>
  )
}

export default App
