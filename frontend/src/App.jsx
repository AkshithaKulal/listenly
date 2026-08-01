import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const apiUrl = (path) => `${API_BASE}${path}`

const EMOTION_COLORS = {
  neutral: '#95a5a6',
  calm: '#3498db',
  happy: '#f1c40f',
  sad: '#2980b9',
  angry: '#e74c3c',
  fear: '#8e44ad',
  disgust: '#27ae60',
  surprise: '#e67e22',
}

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="16" fill="#1C1B1A" />
      <path
        d="M20 32c0-6.6 5.4-12 12-12s12 5.4 12 12"
        stroke="#E8B86D"
        strokeWidth="3.2"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M26 32c0-3.3 2.7-6 6-6s6 2.7 6 6"
        stroke="#FAF8F4"
        strokeWidth="2.8"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="32" cy="40" r="3.2" fill="#C9852A" />
    </svg>
  )
}

function WaveField() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let frame = 0
    let raf = 0

    const resize = () => {
      const parent = canvas.parentElement
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = parent.clientWidth * dpr
      canvas.height = parent.clientHeight * dpr
      canvas.style.width = `${parent.clientWidth}px`
      canvas.style.height = `${parent.clientHeight}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const draw = () => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      ctx.clearRect(0, 0, w, h)
      const lines = 7
      for (let i = 0; i < lines; i++) {
        const yBase = h * (0.18 + i * 0.11)
        ctx.beginPath()
        ctx.strokeStyle = i % 2 === 0 ? 'rgba(28,27,26,0.14)' : 'rgba(201,133,42,0.22)'
        ctx.lineWidth = 1.8
        for (let x = 0; x <= w; x += 3) {
          const amp = 22 + i * 7
          const y =
            yBase +
            Math.sin(x * 0.011 + frame * 0.02 + i) * amp +
            Math.sin(x * 0.0035 + frame * 0.012 + i * 1.4) * (amp * 0.5)
          if (x === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
      frame += 1
      raf = requestAnimationFrame(draw)
    }

    resize()
    draw()
    window.addEventListener('resize', resize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={canvasRef} className="wave-field" aria-hidden="true" />
}

function HeroVisual() {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const emotions = Object.keys(EMOTION_COLORS)

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const ctx = canvas.getContext('2d')
    let frame = 0
    let raf = 0

    const resize = () => {
      const size = Math.min(wrap.clientWidth, 420)
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = size * dpr
      canvas.height = size * dpr
      canvas.style.width = `${size}px`
      canvas.style.height = `${size}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const draw = () => {
      const size = canvas.clientWidth
      if (!size) {
        raf = requestAnimationFrame(draw)
        return
      }
      ctx.clearRect(0, 0, size, size)
      const cx = size / 2
      const cy = size / 2

      for (let r = 0; r < 4; r++) {
        ctx.beginPath()
        ctx.arc(cx, cy, 70 + r * 38 + Math.sin(frame * 0.03 + r) * 4, 0, Math.PI * 2)
        ctx.strokeStyle = r % 2 === 0 ? 'rgba(28,27,26,0.12)' : 'rgba(201,133,42,0.28)'
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      const bars = 28
      for (let i = 0; i < bars; i++) {
        const h =
          18 +
          Math.abs(Math.sin(frame * 0.08 + i * 0.45)) * 48 +
          Math.abs(Math.sin(frame * 0.05 + i * 0.2)) * 18
        const x = cx - (bars * 7) / 2 + i * 7
        const y = cy - h / 2
        ctx.fillStyle = i % 3 === 0 ? '#c9852a' : '#1c1b1a'
        ctx.fillRect(x, y, 4, h)
      }

      frame += 1
      raf = requestAnimationFrame(draw)
    }

    resize()
    draw()
    window.addEventListener('resize', resize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <motion.div
      className="hero-visual"
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="hero-orb" ref={wrapRef}>
        <canvas ref={canvasRef} aria-hidden="true" />
        {emotions.map((emotion, i) => {
          const angle = (i / emotions.length) * Math.PI * 2 - Math.PI / 2
          const radius = 46
          const x = 50 + Math.cos(angle) * radius
          const y = 50 + Math.sin(angle) * radius
          return (
            <span
              key={emotion}
              className="hero-emotion"
              style={{
                left: `${x}%`,
                top: `${y}%`,
                ['--dot']: EMOTION_COLORS[emotion],
                animationDelay: `${i * 0.08}s`,
              }}
            >
              {emotion}
            </span>
          )
        })}
      </div>
      <p className="hero-visual-caption">Eight emotions read from tone, pace, and timbre</p>
    </motion.div>
  )
}

function MiniWave({ file }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!file) return
    let cancelled = false
    const draw = async () => {
      try {
        const buffer = await file.arrayBuffer()
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
        const audio = await audioCtx.decodeAudioData(buffer.slice(0))
        if (cancelled) {
          audioCtx.close()
          return
        }
        const data = audio.getChannelData(0)
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const dpr = Math.min(window.devicePixelRatio || 1, 2)
        const w = canvas.clientWidth
        const h = canvas.clientHeight
        canvas.width = w * dpr
        canvas.height = h * dpr
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        ctx.clearRect(0, 0, w, h)
        ctx.strokeStyle = '#1c1b1a'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        const step = Math.max(1, Math.floor(data.length / w))
        for (let x = 0; x < w; x++) {
          const i = x * step
          const y = h / 2 + data[i] * (h * 0.42)
          if (x === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
        audioCtx.close()
      } catch {
        /* ignore decode errors for unsupported preview */
      }
    }
    draw()
    return () => {
      cancelled = true
    }
  }, [file])

  if (!file) return null
  return (
    <div className="wave-preview">
      <canvas ref={canvasRef} />
    </div>
  )
}

function AudioPreview({ file }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])
  if (!url) return null
  return <audio controls src={url} style={{ width: '100%' }} />
}

function ResultPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="result-panel empty">
        <motion.div
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.4, repeat: Infinity }}
        >
          Listening to the signal…
        </motion.div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="result-panel empty">
        <p>Upload or record speech. Emotion confidence will appear here.</p>
      </div>
    )
  }

  const sorted = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1])

  return (
    <motion.div
      className="result-panel"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <div>
        <div className="result-emotion">
          <span className="label" style={{ color: result.color || '#fff' }}>
            {result.emotion}
          </span>
          <span className="conf">{(result.confidence * 100).toFixed(1)}% confidence</span>
        </div>
        <p style={{ margin: '0.35rem 0 0', opacity: 0.8, fontSize: '0.9rem' }}>
          Predicted from CNN + LSTM on MFCC features
        </p>
      </div>
      <div className="bars">
        {sorted.map(([emotion, prob], i) => (
          <div className="bar-row" key={emotion}>
            <span style={{ textTransform: 'capitalize' }}>{emotion}</span>
            <div className="bar-track">
              <motion.div
                className="bar-fill"
                style={{ background: EMOTION_COLORS[emotion] || '#fff' }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: prob }}
                transition={{ delay: 0.08 * i, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            <span>{(prob * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [recording, setRecording] = useState(false)
  const mediaRef = useRef(null)
  const chunksRef = useRef([])
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(apiUrl('/api/health'))
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'down', model_ready: false }))
  }, [])

  const onFiles = useCallback((list) => {
    const f = list?.[0]
    if (!f) return
    setFile(f)
    setResult(null)
    setError('')
  }, [])

  const analyze = async (audioFile = file) => {
    if (!audioFile) {
      setError('Choose or record an audio file first.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const body = new FormData()
      body.append('file', audioFile, audioFile.name || 'recording.webm')
      const res = await fetch(apiUrl('/api/predict'), { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
    } catch (e) {
      setError(e.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const toggleRecord = async () => {
    if (recording) {
      mediaRef.current?.stop()
      setRecording(false)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const recFile = new File([blob], `recording-${Date.now()}.webm`, { type: 'audio/webm' })
        setFile(recFile)
        setResult(null)
        setError('')
      }
      mediaRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch {
      setError('Microphone access denied or unavailable.')
    }
  }

  return (
    <div className={`app ${recording ? 'recording' : ''}`}>
      <nav className="nav">
        <a className="brand" href="#top">
          <BrandMark />
          <span className="brand-name">Listenly</span>
        </a>
        <div className="nav-links">
          <a href="#top">Home</a>
          <a href="#analyze">Analyze</a>
          <span className="nav-meta">
            <span className={`pulse ${health?.model_ready ? '' : 'off'}`} />
            {health?.model_ready ? 'Model ready' : 'Model offline'}
          </span>
        </div>
      </nav>

      <header className="hero" id="top">
        <WaveField />
        <div className="hero-grid">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          >
            <h1>Listenly</h1>
            <p>
              Soft listening for hard feelings — hear calm, joy, anger, and the quiet spaces between
              the words.
            </p>
            <div className="cta-row">
              <a className="btn btn-primary" href="#analyze">
                Analyze speech
              </a>
              <a className="btn btn-ghost" href="#analyze">
                Upload or record
              </a>
            </div>
            <div className="hero-steps">
              <div>
                <strong>01</strong>
                <span>Upload or record</span>
              </div>
              <div>
                <strong>02</strong>
                <span>Model listens</span>
              </div>
              <div>
                <strong>03</strong>
                <span>Emotion appears</span>
              </div>
            </div>
          </motion.div>
          <HeroVisual />
        </div>
      </header>

      <section className="section" id="analyze">
        <div className="section-head">
          <h2>Read the emotion in a voice</h2>
          <p>Drop a clip, or record a few seconds. Listenly returns an emotion with confidence.</p>
        </div>

        <div className="analyzer">
          <div>
            <div
              className={`dropzone ${dragOver ? 'active' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                onFiles(e.dataTransfer.files)
              }}
            >
              {!file ? (
                <>
                  <div className="drop-icon">♪</div>
                  <h3>Drop audio here</h3>
                  <p className="hint">WAV, MP3, OGG, FLAC, M4A, or WebM</p>
                </>
              ) : (
                <>
                  <div className="file-chip">
                    <div>
                      <strong>{file.name}</strong>
                      <span>{(file.size / 1024).toFixed(1)} KB</span>
                    </div>
                    <button className="btn btn-ghost" type="button" onClick={() => setFile(null)}>
                      Clear
                    </button>
                  </div>
                  <MiniWave file={file} />
                  {file.type.startsWith('audio') && <AudioPreview file={file} />}
                </>
              )}
            </div>

            <div className="actions">
              <button className="btn btn-ghost" type="button" onClick={() => inputRef.current?.click()}>
                Browse files
              </button>
              <button className="btn btn-accent" type="button" onClick={toggleRecord}>
                {recording ? 'Stop recording' : 'Record voice'}
              </button>
              <button
                className="btn btn-primary"
                type="button"
                disabled={loading || !file}
                onClick={() => analyze()}
              >
                {loading ? 'Analyzing…' : 'Predict emotion'}
              </button>
              <input
                ref={inputRef}
                className="hidden-input"
                type="file"
                accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a,.webm"
                onChange={(e) => onFiles(e.target.files)}
              />
            </div>
            <p className={`status-line ${error ? 'error' : ''}`}>{error}</p>
          </div>

          <AnimatePresence mode="wait">
            <ResultPanel
              key={result?.emotion || (loading ? 'load' : 'empty')}
              result={result}
              loading={loading}
            />
          </AnimatePresence>
        </div>
      </section>

      <footer className="footer">
        Listenly · Home & Analyze · CNN+LSTM speech emotion recognition
        <br />
        Built by Akshitha Kulal
      </footer>
    </div>
  )
}
