import React, { useState, useEffect, useRef, useCallback } from 'react';

const BASE_URL = "http://127.0.0.1:8000";

// --- SVG Icons (Premium Minimalist Style) ---
const Icons = {
  Send: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>,
  Pencil: ({ size = 14 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>,
  Trash: ({ size = 14 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>,
  Menu: ({ size = 18 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>,
  More: ({ size = 18 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="1"></circle><circle cx="12" cy="5" r="1"></circle><circle cx="12" cy="19" r="1"></circle></svg>,
  Expand: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>,
  Collapse: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>,
  Check: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>,
  X: ({ size = 16 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>,
  Flight: ({ size = 20 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.2-1.1.6L3 8l6 5.8L7 17l-3-1-1.5 1.5L6 20l2.5 3.5L10 22l-1-3 3.2-2 5.8 6l1.2-.7c.4-.2.7-.6.6-1.1z"/></svg>,
  Hotel: ({ size = 20 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M10 22v-6.57"/><path d="M12 11h.01"/><path d="M12 7h.01"/><path d="M14 15.43V22"/><path d="M15 16a5 5 0 0 0-6 0"/><path d="M16 11h.01"/><path d="M16 7h.01"/><path d="M8 11h.01"/><path d="M8 7h.01"/><rect x="4" y="2" width="16" height="20" rx="2"/></svg>,
  Sun: ({ size = 20 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>,
};

function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/^### (.+)$/gm, '<h3 class="md-h3">$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2 class="md-h2">$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1 class="md-h1">$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,    '<em>$1</em>')
    .replace(/^[\-\*] (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, s => `<ul class="md-ul">${s}</ul>`)
    .replace(/\n\n/g, '</p><p class="md-p">')
    .replace(/^(?!<[hul])(.+)$/gm, '<p class="md-p">$1</p>')
    .replace(/<p class="md-p"><\/p>/g, '');
}

function Spinner() {
  return (
    <svg className="spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
    </svg>
  );
}

const NODE_LABELS = {
  extract_metadata:    { icon: '📋', label: 'Reading message' },
  validator:           { icon: '🛡️', label: 'Validating' },
  master_orchestrator: { icon: '🎯', label: 'Routing intent' },
  cache_check:         { icon: '⚡', label: 'Cache check' },
  master_planner:      { icon: '🗺️', label: 'Planning trip' },
  critic:              { icon: '🔍', label: 'Quality review' },
  hitl_approval:       { icon: '✋', label: 'Awaiting approval' },
  cache_store:         { icon: '💾', label: 'Saving to cache' },
  researcher:          { icon: '🔎', label: 'Researching' },
  preferences_memory:  { icon: '🧠', label: 'Loading prefs' },
  summarizer:          { icon: '📝', label: 'Summarizing' },
};

function SplashScreen({ onEnter }) {
  const [visible, setVisible] = useState(false);
  const [leaving, setLeaving] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 60); return () => clearTimeout(t); }, []);
  const handleEnter = () => { setLeaving(true); setTimeout(() => onEnter(), 700); };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'linear-gradient(135deg, #0a0f1e 0%, #111827 40%, #1a2540 70%, #0d1424 100%)',
      transition: 'opacity 0.7s ease, transform 0.7s ease', opacity: leaving ? 0 : visible ? 1 : 0, transform: leaving ? 'scale(1.04)' : 'scale(1)', overflow: 'hidden',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        .splash-btn { font-family:'Outfit', sans-serif; font-size:15px; font-weight:700; letter-spacing:0.06em; padding:16px 48px; background:linear-gradient(135deg, #0284c7 0%, #38bdf8 100%); color:#fff; border:none; border-radius:50px; cursor:pointer; transition:all 0.2s ease; box-shadow:0 8px 30px rgba(56,189,248,0.35); }
        .splash-btn:hover { filter: brightness(1.12); transform: translateY(-2px) scale(1.02); }
      `}</style>
      <div style={{ position:'relative', zIndex:2, textAlign:'center', display:'flex', flexDirection:'column', alignItems:'center', fontFamily:"'Outfit', sans-serif" }}>
        <div style={{ fontSize:80, marginBottom:4, lineHeight:1, filter:'drop-shadow(0 6px 24px rgba(56,189,248,0.3))' }}>🌍</div>
        <div style={{ marginTop:20, marginBottom:8 }}>
          <div style={{ fontSize:11, fontWeight:700, letterSpacing:'0.25em', color:'rgba(56,189,248,0.7)' }}>LUXURY AI TRAVEL CONCIERGE</div>
          <h1 style={{ fontSize:52, fontWeight:800, letterSpacing:'-0.04em', background:'linear-gradient(135deg, #f0f9ff 0%, #38bdf8 50%, #818cf8 100%)', WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent' }}>marco</h1>
        </div>
        <p style={{ color:'rgba(148,163,184,0.85)', fontSize:15, maxWidth:380, lineHeight:1.7, margin:'0 auto 40px' }}>Design fully personalized itineraries, discover the best routes, and unlock curated experiences — in seconds.</p>
        <button className="splash-btn" onClick={handleEnter}>START CHATTING</button>
      </div>
    </div>
  );
}

export default function App() {
  const [showSplash, setShowSplash]           = useState(true);
  
  // FIX: Load active session from localStorage to prevent loss on refresh
  const [activeSession, setActiveSession]     = useState(() => {
    try {
      return localStorage.getItem('activeSession') || crypto.randomUUID();
    } catch {
      return crypto.randomUUID();
    }
  });

const [sessions, setSessions] = useState(() => {
  try {
    const saved = localStorage.getItem('allSessions');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
});

  const [messages, setMessages]               = useState([]);
  const [hitlPending, setHitlPending]         = useState(false);
  const [prompt, setPrompt]                   = useState('');
  const [feedback, setFeedback]               = useState('');
  const [isLoading, setIsLoading]             = useState(false);
  const [activeNodes, setActiveNodes]         = useState([]);
  const [completedNodes, setCompletedNodes]   = useState([]);
  const [kpi, setKpi]                         = useState({ cacheStatus: 'Cache Miss', cacheTtl: 'Live Feed' });
  const [criticData, setCriticData]           = useState(null);
  const [menuOpen, setMenuOpen]               = useState(false);
  const [sidebarOpen, setSidebarOpen]         = useState(true);
  const [theme, setTheme]                     = useState('light');
  const [animationsOn, setAnimationsOn]       = useState(true);
  const [hasStartedChat, setHasStartedChat]   = useState(false);
  
  const [agentState, setAgentState]           = useState({});
  const [activeCard, setActiveCard]           = useState(null);
  const [panelMode, setPanelMode]             = useState('split');

  const [formDest, setFormDest]               = useState('');
  const [formDates, setFormDates]             = useState('');
  const [formBudget, setFormBudget]           = useState('');
  const [formCitizenship, setFormCitizenship] = useState('');
  const [formAirport, setFormAirport]         = useState('');

  const [editingSession, setEditingSession]   = useState(null);
  const [editValue, setEditValue]             = useState('');
  const [sessionNames, setSessionNames] = useState(() => {
    try { const saved = localStorage.getItem('sessionNames'); return saved ? JSON.parse(saved) : {}; } catch { return {}; }
  });

  const abortRef  = useRef(null);
  const bottomRef = useRef(null);
  const menuRef   = useRef(null);
  const dark = theme === 'dark';

  // FIX: Save active session to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('activeSession', activeSession);
    } catch {}
  }, [activeSession]);

  useEffect(() => { setHasStartedChat(messages.length > 0); }, [messages, activeSession]);

  useEffect(() => {
    const h = e => { if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, activeNodes]);

  const fetchSessions = useCallback(async () => {
  try {
    const r = await fetch(`${BASE_URL}/sessions`);
    const d = await r.json();
    const backendSessions = d.sessions || [];
    setSessions(prev => {
      const merged = [...new Set([...prev, ...backendSessions])];
      try { localStorage.setItem('allSessions', JSON.stringify(merged)); } catch {}
      return merged;
    });
  } catch {}
}, []);

  const fetchState = useCallback(async (sid) => {
    try {
      const r = await fetch(`${BASE_URL}/session/${sid}/state`);
      if (!r.ok) return;
      const data = await r.json();
      const vals = data.values || {};
      
      setAgentState(vals);

      const msgs = (vals.messages || []).map(m => {
        const content = m.content ?? m.kwargs?.content ?? String(m);
        const role = (m.type === 'human' || String(m.id || '').includes('Human')) ? 'user' : 'assistant';
        return { role, content: String(content) };
      }).filter(m => m.content.trim());
      setMessages(msgs);
      if (msgs.length > 0) setHasStartedChat(true);
      setKpi({ cacheStatus: vals.cache_status || 'Cache Miss', cacheTtl: vals.cache_ttl || 'Live Feed' });
      const cd = vals.critic_results || vals.critique_result || null;
      setCriticData(cd?.score != null ? cd : null);
      setHitlPending((data.next || []).length > 0);
    } catch {}
  }, []);

  useEffect(() => {
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
    setIsLoading(false); setActiveNodes([]); setCompletedNodes([]); setCriticData(null); setMessages([]); setHasStartedChat(false); setPanelMode('split');
    fetchSessions(); fetchState(activeSession);
  }, [activeSession]);

const handleNewPlan = () => {
  const newId = crypto.randomUUID();
  setSessions(prev => {
    const updated = [...new Set([...prev, newId])];
    try { localStorage.setItem('allSessions', JSON.stringify(updated)); } catch {}
    return updated;
  });
  setActiveSession(newId);
  setFeedback(''); setPrompt(''); setHasStartedChat(false);
  setPanelMode('split'); setAgentState({}); setActiveCard(null);
};

const handleDelete = async (sid, e) => {
  e.stopPropagation();
  await fetch(`${BASE_URL}/session/${sid}`, { method: 'DELETE' });
  setSessions(prev => {
    const updated = prev.filter(s => s !== sid);
    try { localStorage.setItem('allSessions', JSON.stringify(updated)); } catch {}
    return updated;
  });
  if (sid === activeSession) handleNewPlan(); else fetchSessions();
};

  const handleStartEdit = (sid, e) => { e.stopPropagation(); setEditingSession(sid); setEditValue(sessionNames[sid] || sid.substring(0, 8) + '…'); };

  const handleSaveEdit = async (sid) => {
    setEditingSession(null);
    if (!editValue.trim()) return;
    const finalName = editValue.trim();
    setSessionNames(prev => { const updated = { ...prev, [sid]: finalName }; try { localStorage.setItem('sessionNames', JSON.stringify(updated)); } catch (err) {} return updated; });
    try { await fetch(`${BASE_URL}/session/${sid}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: finalName }) }); } catch (err) {}
  };

  const handleReset = async () => {
    if (!window.confirm('Reset all sessions? This clears the entire checkpoint database.')) return;
    await fetch(`${BASE_URL}/sessions`, { method: 'DELETE' }); setMenuOpen(false); handleNewPlan(); fetchSessions();
  };

  const handleRefresh = () => {
    fetchState(activeSession);
    setMenuOpen(false);
  };

  const handlePrint = () => {
    window.print();
    setMenuOpen(false);
  };

  const executeChat = async (userContent) => {
    setPrompt(''); setMessages(prev => [...prev, { role: 'user', content: userContent }]); setHasStartedChat(true); setIsLoading(true); setActiveNodes([]); setCompletedNodes([]); setCriticData(null);
    const controller = new AbortController(); abortRef.current = controller;
    try {
      const res = await fetch(`${BASE_URL}/chat/stream/${activeSession}?message=${encodeURIComponent(userContent)}`, { signal: controller.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }); const lines = buffer.split('\n'); buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          try {
            const ev = JSON.parse(line.slice(5).trim());
            if (ev.type === 'node') { setActiveNodes([ev.node]); setCompletedNodes(prev => prev.includes(ev.node) ? prev : [...prev, ev.node]); }
            else if (ev.type === 'done') {
              if (ev.reply) setMessages(prev => [...prev, { role: 'assistant', content: ev.reply }]);
              setHitlPending(!!ev.hitl); setIsLoading(false); setActiveNodes([]); await fetchState(activeSession); await fetchSessions();
            } else if (ev.type === 'error') { setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${ev.reply}` }]); setIsLoading(false); setActiveNodes([]); }
          } catch {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setMessages(prev => [...prev, { role: 'assistant', content: 'Connection lost. Please try again.' }]);
      setIsLoading(false); setActiveNodes([]);
    }
  }

  const handleSend = e => {
    if (e) e.preventDefault();
    if (!prompt.trim() || isLoading || hitlPending) return;
    executeChat(prompt.trim());
  };

  const handleFormSubmit = e => {
    if (e) e.preventDefault();
    if (!formDest.trim() || !formCitizenship.trim() || !formAirport.trim() || isLoading || hitlPending) return;
    const content = `Plan a trip to ${formDest.trim()}.${formDates ? ` Dates: ${formDates}.` : ''}${formBudget ? ` Budget: ${formBudget}.` : ''} Citizenship: ${formCitizenship.trim()}. Departure Airport: ${formAirport.trim()}.`;
    executeChat(content);
  }

  const handleHitl = async action => {
    setIsLoading(true); setActiveNodes(['hitl_approval']);
    try {
      const res = await fetch(`${BASE_URL}/session/${activeSession}/resume`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, feedback }) });
      const data = await res.json();
      if (data.reply) setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
      setHitlPending(data.hitl || false); setFeedback('');
    } catch { setMessages(prev => [...prev, { role: 'assistant', content: 'Resume action failed.' }]); } 
    finally { setIsLoading(false); setActiveNodes([]); await fetchState(activeSession); await fetchSessions(); }
  };

  const otherSessions = [...new Set([...sessions, activeSession])].filter(s => s !== activeSession);
  const scoreColor = s => s >= 8 ? '#10b981' : s >= 6 ? '#f59e0b' : '#ef4444';

  const t = dark ? {
    bg:        'transparent',
    surface:   'rgba(20, 24, 33, 0.82)',
    surfaceHi: 'rgba(28, 34, 46, 0.90)',
    border:    'rgba(255,255,255,0.07)',
    borderHi:  'rgba(255,255,255,0.14)',
    text:      '#f3f4f6',
    muted:     '#9ca3af',
    accent:    '#38bdf8',
    accentGlow:'rgba(56, 189, 248, 0.25)',
    green:     '#34d399',
    amber:     '#fbbf24',
    red:       '#f87171',
    sidebarBg: 'rgba(14, 18, 26, 0.88)'
  } : {
    bg:        'transparent',
    surface:   'rgba(252, 250, 247, 0.85)',
    surfaceHi: 'rgba(244, 239, 233, 0.95)',
    border:    'rgba(180, 170, 155, 0.22)',
    borderHi:  'rgba(140, 130, 115, 0.35)',
    text:      '#2d2722',
    muted:     '#7c7267',
    accent:    '#0284c7',
    accentGlow:'rgba(2, 132, 199, 0.2)',
    green:     '#15803d',
    amber:     '#b45309',
    red:       '#b91c1c',
    sidebarBg: 'rgba(247, 243, 237, 0.88)'
  };

  const SidebarBtn = ({ onClick, icon: Icon, title }) => (
    <button title={title} onClick={onClick} style={{ background: 'none', border: 'none', cursor: 'pointer', color: t.muted, padding: '4px', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s' }} onMouseEnter={e => { e.currentTarget.style.color = t.text; e.currentTarget.style.background = dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)'; }} onMouseLeave={e => { e.currentTarget.style.color = t.muted; e.currentTarget.style.background = 'none'; }}>
      <Icon size={14} />
    </button>
  );

  return (
    <>
      {showSplash && <SplashScreen onEnter={() => setShowSplash(false)} />}

      <div style={{ display:'flex', height:'100vh', fontFamily:"'Outfit', sans-serif", color:t.text, overflow:'hidden', position:'relative' }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
          *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
          * { transition-property: background-color, background, color, border-color, box-shadow, fill; transition-duration: 0.4s; transition-timing-function: ease-in-out; }
          .plane-trail, .plane-trail *, .world-dots, .spin, .no-transition { transition: none !important; }

          .bg-canvas { position:fixed; inset:0; z-index:0; background: ${dark ? 'linear-gradient(135deg, #090d16 0%, #121824 40%, #1a2333 70%, #0e131f 100%)' : 'linear-gradient(135deg, #f5efe6 0%, #eae1d4 30%, #fdfbf7 70%, #f2eae1 100%)'}; }
          .bg-canvas::before { content:''; position:absolute; inset:0; background-image: radial-gradient(ellipse 80% 60% at 20% 20%, ${t.accentGlow} 0%, transparent 60%), radial-gradient(ellipse 60% 80% at 80% 80%, ${dark?'rgba(147,197,253,0.05)':'rgba(99,102,241,0.03)'} 0%, transparent 60%); }

          .plane-trail { position:fixed; z-index:1; pointer-events:none; }
          .plane-1 { top:8%;  animation: flyAcross1 22s linear infinite; }
          .plane-2 { top:28%; animation: flyAcross1 35s linear infinite 8s; }
          .plane-3 { top:55%; animation: flyAcross2 28s linear infinite 4s; }
          .plane-4 { top:72%; animation: flyAcross1 40s linear infinite 14s; }
          .plane-5 { top:42%; animation: flyAcross2 18s linear infinite 2s; }
          @keyframes flyAcross1 { 0% { transform: translateX(-150px) translateY(0px) scaleX(1); opacity:0; } 5% { opacity:1; } 95% { opacity:1; } 100% { transform: translateX(110vw) translateY(-30px) scaleX(1); opacity:0; } }
          @keyframes flyAcross2 { 0% { transform: translateX(110vw) translateY(0px) scaleX(-1); opacity:0; } 5% { opacity:1; } 95% { opacity:1; } 100% { transform: translateX(-150px) translateY(20px) scaleX(-1); opacity:0; } }

          .world-dots { position:fixed; inset:0; z-index:1; pointer-events:none; background-image: radial-gradient(circle, ${dark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.04)'} 1px, transparent 1px); background-size: 32px 32px; mask-image: radial-gradient(ellipse 90% 80% at 50% 50%, black 30%, transparent 80%); -webkit-mask-image: radial-gradient(ellipse 90% 80% at 50% 50%, black 30%, transparent 80%); animation: driftGrid 75s linear infinite; }
          @keyframes driftGrid { from { background-position:0 0; } to { background-position:32px 32px; } }
          ${!animationsOn ? '.plane-trail { display: none !important; } .world-dots { animation: none !important; }' : ''}

          ::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:transparent; } ::-webkit-scrollbar-thumb { background:${t.border}; border-radius:2px; }

          .spin { animation: spin 0.9s linear infinite; } @keyframes spin { to { transform:rotate(360deg); } }
          @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
          .fade-up { animation: fadeUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

          .npill { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:20px; font-size:11.5px; font-weight:600; border:1px solid; }
          .npill-active  { background:${t.accentGlow}; border-color:${t.accent}; color:${t.accent}; }
          .npill-done    { background:${dark?'rgba(16,185,129,0.1)':'rgba(21,128,61,0.08)'}; border-color:${t.green}; color:${t.green}; }
          .npill-waiting { background:transparent; border-color:${t.border}; color:${t.muted}; }

          .bubble { padding:18px 22px; border-radius:16px; font-size:14.5px; line-height:1.75; word-break:break-word; border:1px solid; position: relative; z-index: 2; backdrop-filter: blur(10px); }
          .bubble-user  { background: ${dark ? 'rgba(30, 41, 59, 0.7)' : 'rgba(235, 227, 214, 0.85)'}; border-color: ${dark ? 'rgba(255,255,255,0.08)' : 'rgba(2,132,199,0.1)'}; }
          .bubble-agent { background: ${t.surface}; border-color: ${t.border}; }

          .md-h1 { font-size:20px; font-weight:700; margin:18px 0 10px; letter-spacing: -0.01em; } .md-h2 { font-size:17px; font-weight:600; margin:16px 0 8px; color:${t.accent}; } .md-h3 { font-size:15px; font-weight:600; margin:12px 0 6px; } .md-p { margin:6px 0; } .md-ul { padding-left:20px; margin:6px 0; } .md-ul li { margin:4px 0; }

          .glass { background:${t.surface}; border:1px solid ${t.border}; border-radius:14px; position: relative; z-index: 2; backdrop-filter: blur(10px); }

          .sess-row { display:flex; align-items:center; justify-content:space-between; padding:9px 12px; border-radius:9px; cursor:pointer; font-size:12.5px; border:1px solid transparent; }
          .sess-row:hover { background:${dark?'rgba(255,255,255,0.04)':'rgba(2,132,199,0.04)'}; border-color:${t.border}; }
          .sess-row.active { background:${dark?'rgba(56,189,248,0.12)':'rgba(2,132,199,0.06)'}; border-color:${dark?'rgba(56,189,248,0.3)':'rgba(2,132,199,0.2)'}; font-weight: 500; }

          .btn { display:inline-flex; align-items:center; justify-content:center; gap:8px; border:none; border-radius:10px; cursor:pointer; font-family:inherit; font-weight:600; font-size:13px; position: relative; z-index: 2; transition: all 0.15s ease; }
          .btn:disabled { opacity:0.4; cursor:not-allowed; }
          .btn-primary { background:${t.accent}; color: ${dark ? '#111622' : '#ffffff'}; }
          .btn-primary:hover:not(:disabled) { filter:brightness(1.1); box-shadow:0 0 16px ${t.accentGlow}; }
          .btn-ghost { background:transparent; color:${t.muted}; }
          .btn-ghost:hover:not(:disabled) { background:${dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}; color:${t.text}; }
          .btn-outline { background:transparent; color:${t.text}; border:1px solid ${t.border}; }
          .btn-outline:hover:not(:disabled) { background:${dark ? 'rgba(255, 255, 255, 0.03)' : 'rgba(0, 0, 0, 0.02)'}; border-color:${t.borderHi}; }
          
          /* Modals Buttons */
          .btn-green { background:${t.green}; color:#fff; }
          .btn-amber { background:${dark?'rgba(251,191,36,0.12)':'rgba(180,83,9,0.08)'}; color:${t.amber}; border:1px solid ${dark?'rgba(251,191,36,0.25)':'rgba(180,83,9,0.2)'}; }
          .btn-red { background:${dark?'rgba(248,113,113,0.08)':'rgba(185,28,28,0.06)'}; color:${t.red}; border:1px solid ${dark?'rgba(248,113,113,0.2)':'rgba(185,28,28,0.18)'}; }

          .input-wrap { display:flex; align-items:center; gap:12px; padding:12px 16px; background:${t.surfaceHi}; border:1px solid ${t.border}; border-radius:14px; transition:border-color 0.2s, box-shadow 0.2s; }
          .input-wrap:focus-within { border-color:${t.accent}; box-shadow:0 4px 20px ${t.accentGlow}; }
          .input-field { flex:1; background:none; border:none; outline:none; color:${t.text}; font-family:inherit; font-size:14.5px; }
          .input-field::placeholder { color:${t.muted}; }

          .form-input { width: 100%; padding: 12px 14px; background: ${dark ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)'}; border: 1px solid ${t.border}; border-radius: 10px; color: ${t.text}; font-family: inherit; font-size: 14px; outline: none; transition: border-color 0.2s; }
          .form-input:focus { border-color: ${t.accent}; }

          .dynamic-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
          .dynamic-card { padding: 14px 16px; border-radius: 12px; border: 1px solid ${t.border}; background: ${t.surfaceHi}; display: flex; flex-direction: column; gap: 6px; transition: transform 0.2s; cursor: pointer; }
          .dynamic-card:hover { transform: translateY(-2px); border-color: ${t.accent}; box-shadow: 0 10px 30px ${t.accentGlow}; }

          .menu-item { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; font-size:13px; cursor:pointer; }
          .menu-item:hover { background:${dark?'rgba(255,255,255,0.04)':'rgba(2,132,199,0.04)'}; }
          .menu-danger { color:${t.red}; }

          @media print {
            html, body, #root, div[style*="display:flex"] { height: auto !important; overflow: visible !important; display: block !important; }
            .no-print { display:none !important; }
            div[style*="flex:1"] { overflow: visible !important; height: auto !important; max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
            .bubble { border:1px solid #ddd !important; background:#fafafa !important; color:#000 !important; page-break-inside:avoid; }
            .bg-canvas, .world-dots, .plane-trail { display:none !important; }
          }
        `}</style>

        {/* ── Background layers ── */}
        <div className="bg-canvas" />
        <div className="world-dots" />

        {animationsOn && [
          { cls:'plane-1', size:90,  color: t.accent, opacity: dark?0.35:0.25 },
          { cls:'plane-2', size:55,  color: t.accent, opacity: dark?0.22:0.14 },
          { cls:'plane-3', size:110, color: t.accent, opacity: dark?0.30:0.20 },
          { cls:'plane-4', size:40,  color: t.accent, opacity: dark?0.25:0.12 },
          { cls:'plane-5', size:75,  color: t.accent, opacity: dark?0.28:0.18 },
        ].map(({ cls, size, color, opacity }) => (
          <div key={cls} className={`plane-trail ${cls}`} style={{ opacity }}>
            <svg width={size} height={size*0.45} viewBox="0 0 80 36" fill={color}>
              <path d="M76 18L4 2l6 14-6 2 6 14 72-14z"/>
              <path d="M10 16l-4 8 20-4" opacity="0.5"/>
            </svg>
          </div>
        ))}

        {/* ── Sidebar ── */}
        <div className="no-print" style={{
          width: sidebarOpen ? 260 : 0, minWidth: sidebarOpen ? 260 : 0,
          overflow:'hidden', transition:'width 0.3s cubic-bezier(0.16, 1, 0.3, 1), min-width 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
          borderRight:`1px solid ${t.border}`, background: t.sidebarBg, backdropFilter: 'blur(15px)', display:'flex', flexDirection:'column', position:'relative', zIndex:10,
        }}>
            <div style={{ padding:'20px 16px', width:260, display:'flex', flexDirection:'column', height:'100%' }}>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:24 }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ fontSize:16, fontWeight:800, color:t.text, letterSpacing:'-0.02em', textTransform: 'LOWERCASE' }}>marco</span>
              </div>
              <button className="btn btn-ghost" style={{ padding:'6px' }} onClick={() => setSidebarOpen(false)}><Icons.Menu size={16}/></button>
            </div>

            <button className="btn btn-outline" style={{ padding: '12px 14px', marginBottom: 24, width: '100%', fontSize: '13px' }} onClick={handleNewPlan}>
              + Start New Session
            </button>

            <div style={{ marginBottom:20 }}>
              <div style={{ fontSize:10, fontWeight:700, color:t.muted, letterSpacing:'0.09em', textTransform:'uppercase', marginBottom:8 }}>Active Journey</div>
              <div className="sess-row active" style={{ cursor:'default' }}>
                <div style={{ display:'flex', alignItems:'center', gap:7, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', flex: 1 }}>
                  <span style={{ width:6, height:6, borderRadius:'50%', background:t.accent, flexShrink:0 }} />
                  {editingSession === activeSession ? (
                    <input autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onBlur={() => handleSaveEdit(activeSession)} onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(activeSession); if (e.key === 'Escape') setEditingSession(null); }} onClick={e => e.stopPropagation()} style={{ background: 'transparent', border: `1px solid ${t.accent}`, color: t.text, outline: 'none', borderRadius: 4, padding: '2px 6px', fontSize: 'inherit', width: '100%', fontFamily: 'inherit' }} />
                  ) : (
                    <span style={{ overflow:'hidden', textOverflow:'ellipsis' }}>{sessionNames[activeSession] || activeSession.substring(0,8) + '…'}</span>
                  )}
                </div>
                {!editingSession && (
                  <div style={{ display:'flex', alignItems:'center', gap:2, flexShrink:0 }}>
                    <SidebarBtn onClick={e => handleStartEdit(activeSession, e)} icon={Icons.Pencil} title="Rename" />
                    <SidebarBtn onClick={e => handleDelete(activeSession, e)} icon={Icons.Trash} title="Delete" />
                  </div>
                )}
              </div>
            </div>

            {otherSessions.length > 0 && (
              <div style={{ flex:1, overflow:'hidden', display:'flex', flexDirection:'column' }}>
                <div style={{ fontSize:10, fontWeight:700, color:t.muted, letterSpacing:'0.09em', textTransform:'uppercase', marginBottom:8 }}>Saved Trips</div>
                <div style={{ overflowY:'auto', flex:1, display:'flex', flexDirection:'column', gap:4 }}>
                  {otherSessions.map(s => (
                    <div key={s} className="sess-row" onClick={() => setActiveSession(s)}>
                      <div style={{ display:'flex', alignItems:'center', gap:7, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', flex: 1 }}>
                        {editingSession === s ? (
                          <input autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onBlur={() => handleSaveEdit(s)} onKeyDown={e => { if (e.key === 'Enter') handleSaveEdit(s); if (e.key === 'Escape') setEditingSession(null); }} onClick={e => e.stopPropagation()} style={{ background: 'transparent', border: `1px solid ${t.accent}`, color: t.text, outline: 'none', borderRadius: 4, padding: '2px 6px', fontSize: 'inherit', width: '100%', fontFamily: 'inherit' }} />
                        ) : (
                          <span style={{ color:t.muted, overflow:'hidden', textOverflow:'ellipsis' }}>{sessionNames[s] || s.substring(0,8) + '…'}</span>
                        )}
                      </div>
                      {editingSession !== s && (
                        <div style={{ display:'flex', alignItems:'center', gap:2, flexShrink:0 }}>
                          <SidebarBtn onClick={e => handleStartEdit(s, e)} icon={Icons.Pencil} title="Rename" />
                          <SidebarBtn onClick={e => handleDelete(s, e)} icon={Icons.Trash} title="Delete" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Split Screen Container ── */}
        <div style={{ flex:1, display:'flex', overflow:'hidden', position:'relative', zIndex:5 }}>
          
          {/* ── LEFT PANEL: Conversational UI ── */}
          <div className="left-panel" style={{ 
            width: panelMode === 'left' ? '100%' : panelMode === 'split' ? '45%' : '0', 
            minWidth: panelMode === 'right' ? '0' : '400px',
            opacity: panelMode === 'right' ? 0 : 1,
            display: 'flex', flexDirection: 'column', 
            borderRight: panelMode !== 'right' ? `1px solid ${t.border}` : 'none', 
            background: dark ? 'rgba(10, 15, 24, 0.4)' : 'rgba(250, 248, 245, 0.4)',
            transition: 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
            overflow: 'hidden'
          }}>
            
            {/* Header Left */}
            <div style={{ padding: '16px 24px', borderBottom: `1px solid ${t.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', backdropFilter: 'blur(20px)', height: '65px', minHeight: '65px', position: 'relative', zIndex: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {!sidebarOpen && <button className="btn btn-ghost" style={{ padding:'6px' }} onClick={() => setSidebarOpen(true)}><Icons.Menu size={16}/></button>}
                <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0, color: t.text, whiteSpace: 'nowrap' }}>Where to today?</h2>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button className="btn btn-ghost" onClick={() => setPanelMode(panelMode === 'left' ? 'split' : 'left')} style={{ padding: '6px' }} title={panelMode === 'left' ? "Restore Split" : "Maximize Chat"}>
                  {panelMode === 'left' ? <Icons.Collapse size={16}/> : <Icons.Expand size={16}/>}
                </button>
                
                <div ref={menuRef} style={{ position:'relative' }}>
                  <button className="btn btn-ghost" style={{ padding:'6px' }} onClick={() => setMenuOpen(v => !v)}><Icons.More size={18}/></button>
                  {menuOpen && (
                    <div style={{ position:'absolute', right:0, top:'calc(100% + 8px)', width:210, background:t.surfaceHi, border:`1px solid ${t.border}`, borderRadius:12, boxShadow:`0 12px 40px ${dark?'rgba(0,0,0,0.5)':'rgba(0,0,0,0.1)'}`, overflow:'hidden', zIndex:100 }}>
                      <div className="menu-item" onClick={handleRefresh}><span>Refresh Session</span></div>
                      <div style={{ height:1, background:t.border }} />
                      <div className="menu-item" onClick={handlePrint}><span>Print Itinerary</span></div>
                      <div style={{ height:1, background:t.border }} />
                      <div className="menu-item menu-danger" onClick={handleReset}><span>Erase All Data</span></div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Chat History */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: 20 }}>
              {messages.map((msg, idx) => (
                <div key={idx} className="fade-up" style={{ display:'flex', flexDirection:'column', gap:6 }}>
                  <span style={{ fontSize:10, fontWeight:700, color:t.muted, textTransform:'lowercase', letterSpacing:'0.08em' }}>
                    {msg.role === 'user' ? 'GUEST' : 'marco'}
                  </span>
                  <div className={`bubble ${msg.role==='user'?'bubble-user':'bubble-agent'}`}>
                    {msg.role === 'assistant' ? <div dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} /> : msg.content}
                  </div>
                </div>
              ))}

              {/* Status Pills */}
              {isLoading && (
                <div className="fade-up" style={{ padding: '18px', background: t.surface, borderRadius: '16px', border: `1px solid ${t.border}` }}>
                   <div style={{ display:'flex', alignItems:'center', gap:8, fontSize:13, fontWeight:600, color:t.accent, marginBottom: 14 }}>
                    <Spinner /> Orchestrating Itinerary...
                  </div>
                  <div style={{ display:'flex', flexWrap:'wrap', gap:8 }}>
                    {Object.keys(NODE_LABELS).filter(node => activeNodes.includes(node) || completedNodes.includes(node)).map(node => {
                      const isActive = activeNodes.includes(node);
                      const isDone   = completedNodes.includes(node) && !isActive;
                      const { icon, label } = NODE_LABELS[node];
                      return (
                        <span key={node} className={`npill ${isActive?'npill-active':isDone?'npill-done':'npill-waiting'}`}>
                          {isActive ? <Spinner /> : icon} {label}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Chat Input */}
            <div style={{ padding: '16px 24px', borderTop: `1px solid ${t.border}`, background: t.surfaceHi }}>
              <form onSubmit={handleSend}>
                <div className="input-wrap" style={{ borderRadius: '24px', padding: '8px 12px 8px 18px' }}>
                  <input className="input-field" type="text" value={prompt} onChange={e => setPrompt(e.target.value)} disabled={isLoading || hitlPending} placeholder="Describe your dream destination..." />
                  <button type="submit" className="btn btn-primary" disabled={isLoading || hitlPending || !prompt.trim()} style={{ padding:'10px', borderRadius: '50%' }}>
                    {isLoading ? <Spinner /> : <Icons.Send size={18} />}
                  </button>
                </div>
              </form>
            </div>
          </div>


          {/* ── RIGHT PANEL: Dynamic Content Cards & HITL ── */}
          <div className="right-panel" style={{ 
            flex: panelMode === 'left' ? 'none' : 1, 
            width: panelMode === 'left' ? '0' : 'auto',
            opacity: panelMode === 'left' ? 0 : 1,
            position: 'relative', display: 'flex', flexDirection: 'column', 
            background: dark ? 'rgba(18, 24, 38, 0.4)' : 'rgba(240, 236, 228, 0.4)',
            transition: 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
            overflow: 'hidden'
          }}>
            
            {/* Header Right */}
            <div style={{ padding: '16px 24px', borderBottom: `1px solid ${t.border}`, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 16, backdropFilter: 'blur(20px)', height: '65px', minHeight: '65px', position: 'relative', zIndex: 10 }}>
              
              <div style={{ display: 'flex', gap: 16, fontSize: 12, fontWeight: 600, color: t.muted, marginRight: 'auto' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: kpi.cacheStatus==='hit'?t.green:t.amber }}></span> {kpi.cacheStatus}</span>
                <span>⏱️ {kpi.cacheTtl}</span>
              </div>

              {/* Theme & Animation Toggles */}
              <div style={{ display:'flex', gap:2, background:dark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.04)', borderRadius:8, padding:3 }}>
                <button className="btn" onClick={() => setAnimationsOn(!animationsOn)} title="Toggle Animations" style={{ padding:'4px 8px', fontSize:12, borderRadius:6, background: animationsOn ? t.accent : 'transparent', color: animationsOn ? '#ffffff' : t.muted }}>✈️</button>
                <div style={{ width: 1, background: t.border, margin: '2px 4px' }} />
                <button className="btn" onClick={() => setTheme('light')} style={{ padding:'4px 10px', fontSize:11, fontWeight:600, borderRadius:6, background: !dark ? t.accent : 'transparent', color: !dark ? '#ffffff' : t.muted }}>L</button>
                <button className="btn" onClick={() => setTheme('dark')} style={{ padding:'4px 10px', fontSize:11, fontWeight:600, borderRadius:6, background: dark ? t.accent : 'transparent', color: dark ? '#111622' : t.muted }}>D</button>
              </div>
            </div>

            {/* Scrollable Content Right */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '32px' }}>
              
              {/* State 1: Empty Chat -> Show "Get Started" Form */}
              {!hasStartedChat && (
                <div className="fade-up" style={{ maxWidth: 800, margin: '0 auto' }}>
                  <div className="glass" style={{ padding: 24, borderRadius: 20, marginBottom: 16 }}>
                    <h3 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: t.text }}>Plan a New Journey</h3>
                    <p style={{ color: t.muted, fontSize: 14, marginBottom: 20 }}>Enter your basic parameters to instantly generate a full itinerary.</p>
                    
                    <form onSubmit={handleFormSubmit}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px', marginBottom: 20 }}>
                        <div>
                          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: t.muted, marginBottom: 6 }}>DESTINATION</label>
                          <input className="form-input" placeholder="e.g. Paris, France" value={formDest} onChange={e => setFormDest(e.target.value)} required />
                        </div>
                        <div>
                          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: t.muted, marginBottom: 6 }}>DEPARTURE AIRPORT</label>
                          <input className="form-input" placeholder="e.g. TLV" value={formAirport} onChange={e => setFormAirport(e.target.value)} required />
                        </div>
                        <div>
                          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: t.muted, marginBottom: 6 }}>CITIZENSHIP</label>
                          <input className="form-input" placeholder="e.g. Israeli" value={formCitizenship} onChange={e => setFormCitizenship(e.target.value)} required />
                        </div>
                        <div>
                          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: t.muted, marginBottom: 6 }}>DATES / DURATION</label>
                          <input className="form-input" placeholder="e.g. Oct 12-18 or '5 Days'" value={formDates} onChange={e => setFormDates(e.target.value)} />
                        </div>
                        <div style={{ gridColumn: '1 / -1' }}>
                          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: t.muted, marginBottom: 6 }}>BUDGET</label>
                          <input className="form-input" placeholder="e.g. $3000" value={formBudget} onChange={e => setFormBudget(e.target.value)} />
                        </div>
                      </div>
                      <button type="submit" className="btn btn-primary" style={{ padding: '12px 24px', width: '100%' }}>Generate Itinerary</button>
                    </form>
                  </div>
                </div>
              )}

              {/* State 2: Active Chat -> Show Dynamic UI Grid */}
              {hasStartedChat && (
                <div className="fade-up" style={{ maxWidth: 1000, margin: '0 auto' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                    <h3 style={{ fontSize: 20, fontWeight: 700, color: t.text }}>Itinerary Components</h3>
                    {criticData && (
                      <span style={{ padding: '4px 12px', borderRadius: 20, fontSize: 12, fontWeight: 700, background: `${scoreColor(criticData.score)}15`, color: scoreColor(criticData.score), border: `1px solid ${scoreColor(criticData.score)}40` }}>
                        Critic Score: {criticData.score}/10
                      </span>
                    )}
                  </div>

                  {/* Cards */}
                  <div className="dynamic-grid">
                    
                    {/* Flight Card */}
                    <div className="dynamic-card" onClick={() => setActiveCard('fetch_flights')}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, color: t.accent }}>
                        <div style={{ background: t.accentGlow, padding: 6, borderRadius: 8 }}><Icons.Flight size={16} /></div>
                        <span style={{ fontWeight: 600, fontSize: 14, color: t.text }}>Flights</span>
                      </div>
                      <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.4, flex: 1 }}>
                        Click to view flight pathways and routing structures.
                      </div>
                      <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${t.border}`, fontSize: 11, fontWeight: 600 }}>
                        Status: {agentState?.planner_task_results?.fetch_flights ? 'Completed' : 'Pending'}
                      </div>
                    </div>

                    {/* Hotel Card */}
                    <div className="dynamic-card" onClick={() => setActiveCard('fetch_hotels')}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, color: t.amber }}>
                        <div style={{ background: `${t.amber}22`, padding: 6, borderRadius: 8 }}><Icons.Hotel size={16} /></div>
                        <span style={{ fontWeight: 600, fontSize: 14, color: t.text }}>Accommodations</span>
                      </div>
                      <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.4, flex: 1 }}>
                        Click to view primary destination nodes and hotel data.
                      </div>
                      <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${t.border}`, display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 600 }}>
                        <span>Status:</span>
                        <span>{agentState?.planner_task_results?.fetch_hotels ? 'Completed' : 'Pending'}</span>
                      </div>
                    </div>

                    {/* Environment Card */}
                    <div className="dynamic-card" onClick={() => setActiveCard('fetch_weather')}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, color: t.green }}>
                        <div style={{ background: `${t.green}22`, padding: 6, borderRadius: 8 }}><Icons.Sun size={16} /></div>
                        <span style={{ fontWeight: 600, fontSize: 14, color: t.text }}>Environment</span>
                      </div>
                      <div style={{ fontSize: 12, color: t.muted, lineHeight: 1.4, flex: 1 }}>
                        Click to view weather matrix and local events.
                      </div>
                      <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${t.border}`, fontSize: 11, fontWeight: 600 }}>
                        Status: {agentState?.planner_task_results?.fetch_weather ? 'Completed' : 'Pending'}
                      </div>
                    </div>

                  </div>
                </div>
              )}

            </div>
{/* ── CARD DETAILS MODAL ── */}
            {activeCard && (
              <div className="fade-up" style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                background: dark ? 'rgba(10, 15, 24, 0.85)' : 'rgba(250, 248, 245, 0.85)',
                backdropFilter: 'blur(12px)', zIndex: 110,
                display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40
              }}>
                <div style={{
                  background: t.surfaceHi, border: `1px solid ${t.borderHi}`, borderRadius: 24,
                  padding: '32px 32px 40px', maxWidth: 650, width: '100%', maxHeight: '85vh', overflowY: 'auto',
                  boxShadow: `0 24px 60px ${dark?'rgba(0,0,0,0.6)':'rgba(0,0,0,0.1)'}`, position: 'relative'
                }}>
                  <button onClick={() => setActiveCard(null)} style={{ position: 'absolute', top: 24, right: 24, background: dark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)', border: 'none', color: t.muted, cursor: 'pointer', padding: 8, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s' }} onMouseEnter={e => e.currentTarget.style.background = dark?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.1)'} onMouseLeave={e => e.currentTarget.style.background = dark?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.05)'}>
                    <Icons.X size={18} />
                  </button>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
                    <div style={{ 
                      background: activeCard === 'fetch_flights' ? t.accentGlow : activeCard === 'fetch_hotels' ? `${t.amber}22` : `${t.green}22`, 
                      color: activeCard === 'fetch_flights' ? t.accent : activeCard === 'fetch_hotels' ? t.amber : t.green, 
                      padding: 10, borderRadius: 12 
                    }}>
                       {activeCard === 'fetch_flights' ? <Icons.Flight size={24} /> : activeCard === 'fetch_hotels' ? <Icons.Hotel size={24} /> : <Icons.Sun size={24} />}
                    </div>
                    <h2 style={{ fontSize: 24, fontWeight: 700, color: t.text, margin: 0 }}>
                      {{ fetch_flights: 'Flights', fetch_hotels: 'Accommodations', fetch_weather: 'Environment & Weather' }[activeCard] || activeCard} Details
                    </h2>
                  </div>

                  <div style={{ color: t.text, fontSize: 14, lineHeight: 1.7 }}>
                    {(() => {
                      const rawData = agentState?.planner_task_results?.[activeCard] 
                        || agentState?.final_plan?.[activeCard] 
                        || agentState?.planner_structured_results?.[activeCard];

                      if (!rawData) return <div style={{ padding: 20, textAlign: 'center', color: t.muted, background: t.surface, borderRadius: 16, border: `1px dashed ${t.border}` }}>No detailed information available yet.</div>;

                      let data = rawData;
                      if (typeof rawData === 'string') {
                        try { data = JSON.parse(rawData); } 
                        catch (e) { return <div className="glass" style={{ padding: 20 }} dangerouslySetInnerHTML={{ __html: renderMarkdown(rawData) }} />; }
                      }

                      // If data is an Array (e.g. Flights list)
                      if (Array.isArray(data)) {
                        return (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            {data.map((item, i) => (
                              <div key={i} style={{ padding: '20px', display: 'flex', flexWrap: 'wrap', gap: '20px 32px', background: t.surface, border: `1px solid ${t.border}`, borderRadius: 16 }}>
                                {typeof item === 'object' && item !== null ? (
                                  Object.entries(item).map(([k, v]) => (
                                    <div key={k} style={{ display: 'flex', flexDirection: 'column' }}>
                                      <span style={{ fontSize: 11, textTransform: 'uppercase', color: t.muted, fontWeight: 700, letterSpacing: '0.05em', marginBottom: 4 }}>
                                        {k.replace(/_/g, ' ')}
                                      </span>
                                      <span style={{ fontSize: 16, fontWeight: 600, color: k.toLowerCase() === 'price' ? t.green : t.text }}>
                                        {k.toLowerCase() === 'price' && typeof v === 'number' ? `$${v}` : typeof v === 'object' ? JSON.stringify(v) : String(v)}
                                      </span>
                                    </div>
                                  ))
                                ) : (
                                  <span style={{ color: t.text }}>{String(item)}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        );
                      }

                      // If data is a single Object
                      if (typeof data === 'object' && data !== null) {
                        return (
                          <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 16, padding: '20px', display: 'flex', flexWrap: 'wrap', gap: '20px 32px' }}>
                            {Object.entries(data).map(([k, v]) => (
                              <div key={k} style={{ display: 'flex', flexDirection: 'column' }}>
                                <span style={{ fontSize: 11, textTransform: 'uppercase', color: t.muted, fontWeight: 700, letterSpacing: '0.05em', marginBottom: 4 }}>
                                  {k.replace(/_/g, ' ')}
                                </span>
                                <span style={{ fontSize: 16, fontWeight: 600, color: t.text }}>
                                  {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      }

                      // Fallback for simple strings/primitives
                      return <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 14, background: t.surface, padding: 20, borderRadius: 16 }}>{String(data)}</div>;
                    })()}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}