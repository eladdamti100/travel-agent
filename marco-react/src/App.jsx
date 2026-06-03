import React, { useState, useEffect, useRef } from 'react';

const BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [activeSession, setActiveSession] = useState(crypto.randomUUID());
  const [sessions, setSessions] = useState([]);
  const [messagesLog, setMessagesLog] = useState([]);
  const [hitlPending, setHitlPending] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [feedback, setFeedback] = useState("");
  
  const [kpiData, setKpiData] = useState({ cacheStatus: "Cache Miss", cacheTtl: "Live Feed" });
  const [criticData, setCriticData] = useState(null);
  const [isCriticExpanded, setIsCriticExpanded] = useState(false);
  const [showStreamlitMenu, setShowStreamlitMenu] = useState(false);
  const [theme, setTheme] = useState('dark');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const [isLoading, setIsLoading] = useState(false);
  const [progressLogs, setProgressLogs] = useState("");
  
  const messagesEndRef = useRef(null);
  // רפרנס לשמירת צינור הסטרימינג האקטיבי כדי שנוכל להרוג אותו במעבר סשן
  const eventSourceRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messagesLog, progressLogs]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${BASE_URL}/sessions`);
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (e) {
      console.error("Failed to fetch sessions");
    }
  };

  const fetchState = async () => {
    try {
      const res = await fetch(`${BASE_URL}/session/${activeSession}/state`);
      if (res.ok) {
        const data = await res.json();
        const vals = data.values || {};
        
        const msgs = vals.messages || [];
        const formattedMsgs = msgs.map(msg => {
          const content = msg.content || (msg.kwargs && msg.kwargs.content) || msg;
          const role = (msg.type === "human" || String(msg.id).includes("Human")) ? "user" : "assistant";
          return { role, content };
        }).filter(m => m.content);
        setMessagesLog(formattedMsgs);

        setKpiData({
          cacheStatus: vals.cache_status || "Cache Miss",
          cacheTtl: vals.cache_ttl || "Live Feed"
        });

        const critic = vals.critic_results || vals.critique_result || null;
        setCriticData(critic && critic.score ? critic : null);
        setHitlPending(data.next && data.next.length > 0);
      }
    } catch (e) {
      console.error("Failed to fetch state");
    }
  };

  // בכל פעם שמחליפים סשן - אנחנו קודם כל הורגים את צינור ההקשבה הישן ומאפסים לוגיקה!
  useEffect(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsLoading(false);
    setProgressLogs("");
    setCriticData(null);
    
    fetchSessions();
    fetchState();
  }, [activeSession]);

  const handleStartNewPlan = () => {
    setActiveSession(crypto.randomUUID());
    setMessagesLog([]);
    setHitlPending(false);
    setKpiData({ cacheStatus: "Cache Miss", cacheTtl: "Live Feed" });
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await fetch(`${BASE_URL}/session/${sessionId}`, { method: 'DELETE' });
      if (sessionId === activeSession) {
        handleStartNewPlan();
      } else {
        fetchSessions();
      }
    } catch (err) {
      alert("Failed to delete session.");
    }
  };

  const handleRerun = async () => {
    setIsLoading(true);
    await fetchSessions();
    await fetchState();
    setIsLoading(false);
    setShowStreamlitMenu(false);
  };

  const handleClearAllSessions = async () => {
    if (window.confirm("Are you sure you want to clear the entire global cache database from disk?")) {
      try {
        await fetch(`${BASE_URL}/sessions`, { method: 'DELETE' });
        handleStartNewPlan();
        fetchSessions();
        setShowStreamlitMenu(false);
      } catch (err) {
        alert("Clear cache failed.");
      }
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || hitlPending) return;

    // הגנה: אם יש צינור פתוח, נסגור אותו
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const userMsg = { role: "user", content: prompt };
    setMessagesLog(prev => [...prev, userMsg]);
    setPrompt("");
    setIsLoading(true);
    setProgressLogs("");
    setCriticData(null);

    const sseUrl = `${BASE_URL}/chat/stream/${activeSession}?message=${encodeURIComponent(userMsg.content)}`;
    const eventSource = new EventSource(sseUrl);
    eventSourceRef.current = eventSource; // שמירה ברפרנס הגלובלי

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "node") {
          setProgressLogs(prev => prev + `⚙️ Active Agent Node: **${data.node}**\n`);
        } else if (data.type === "done") {
          if (data.reply) setMessagesLog(prev => [...prev, { role: "assistant", content: data.reply }]);
          setHitlPending(data.hitl);
          setIsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
          fetchState();
          fetchSessions();
        } else if (data.type === "error") {
          setMessagesLog(prev => [...prev, { role: "assistant", content: `⚠️ ${data.reply}` }]);
          setIsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
        }
      } catch (err) {
         console.error(err);
      }
    };

    eventSource.onerror = () => {
      // מונע הקפצה של שגיאה אם אנחנו אלו שסגרנו את הסשן במעבר יזום
      if (eventSourceRef.current) {
        setMessagesLog(prev => [...prev, { role: "assistant", content: "⚠️ Connection to server lost." }]);
        setIsLoading(false);
        eventSource.close();
        eventSourceRef.current = null;
      }
    };
  };

  const handleHitlAction = async (action) => {
    if (action === "edit" && !feedback.trim()) {
      alert("Please enter revision notes before requesting edits.");
      return;
    }
    
    setIsLoading(true);
    setProgressLogs(`🔄 Resuming graph via Human Action: ${action.toUpperCase()}...\n`);
    
    try {
      const res = await fetch(`${BASE_URL}/session/${activeSession}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, feedback })
      });
      
      if (res.ok) {
        const data = await res.json();
        if (data.reply) {
          setMessagesLog(prev => [...prev, { role: "assistant", content: data.reply }]);
        }
        setHitlPending(data.hitl || false);
        setFeedback("");
      }
    } catch (err) {
      alert("Resume action failed.");
    } {
      setIsLoading(false);
      fetchState(); 
      fetchSessions();
    }
  };

  const otherSessions = Array.from(new Set([...sessions, activeSession])).filter(s => s !== activeSession);

  const isDark = theme === 'dark';
  const mainBg = isDark ? 'bg-[#0e1117] text-[#fafafa]' : 'bg-[#ffffff] text-[#313543]';
  const sidebarBg = isDark ? 'bg-[#131720] border-[#1f242e]' : 'bg-[#f0f2f6] border-[#e6e8ed]';
  const sidebarTxt = isDark ? 'text-zinc-200' : 'text-zinc-800';
  const cardBg = isDark ? 'bg-[#262730] border-[#313543]' : 'bg-[#f0f2f6] border-[#e6e8ed]';
  const cardTxt = isDark ? 'text-zinc-100' : 'text-zinc-800';
  const inputBg = isDark ? 'bg-[#262730] border-[#313543]' : 'bg-[#ffffff] border-[#e6e8ed]';
  const titleColor = isDark ? 'text-white' : 'text-[#313543]';

  return (
    <div className={`flex h-screen font-sans antialiased selection:bg-zinc-700 ${mainBg}`}>
      
      <style>{`
        @media print {
          html, body, #root, .flex, .print-area {
            height: auto !important;
            min-height: auto !important;
            overflow: visible !important;
            background: #ffffff !important;
            color: #000000 !important;
            display: block !important;
          }
          .no-print { display: none !important; }
          .print-area { width: 100% !important; position: static !important; padding: 20px !important; }
          .print-content-wrapper { height: auto !important; overflow: visible !important; display: block !important; padding-bottom: 0 !important; }
          .print-bubble { 
            background: #f4f4f5 !important; 
            color: #000000 !important; 
            border: 1px solid #d4d4d8 !important; 
            max-width: 100% !important; 
            box-shadow: none !important;
            margin-bottom: 20px !important;
            padding: 15px !important;
            border-radius: 8px !important;
            display: block !important;
            page-break-inside: avoid;
          }
          .print-title { color: #000000 !important; font-size: 26px !important; margin-bottom: 15px !important; }
          .print-kpi { display: flex !important; justify-content: space-between !important; margin-bottom: 20px !important; border-bottom: 1px solid #ccc !important; padding-bottom: 15px !important; }
          .print-kpi-val { color: #000000 !important; font-size: 16px !important; }
        }
      `}</style>
      
      {/* Sidebar */}
      <div className={`no-print flex flex-col border-r transition-all duration-300 ease-in-out ${sidebarBg} ${
        isSidebarOpen ? 'w-[290px]' : 'w-0 overflow-hidden border-r-0'
      }`}>
        <div className="p-4 flex justify-end">
          <span onClick={() => setIsSidebarOpen(false)} className="text-zinc-500 text-xs font-bold hover:text-zinc-300 cursor-pointer p-1 transition">«</span>
        </div>
        <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-6 w-[290px]">
          <div className={`flex items-center gap-2 text-sm font-semibold ${isDark ? 'text-zinc-200' : 'text-zinc-700'}`}>
            <span>📁</span> Saved Itineraries
          </div>
          <button 
            onClick={handleStartNewPlan}
            className={`w-full bg-transparent hover:bg-zinc-500/10 rounded-md py-1.5 px-3 text-xs font-medium transition border ${
              isDark ? 'text-zinc-200 border-[#313543]' : 'text-zinc-700 border-zinc-300'
            }`}
          >
            <span className="text-purple-500 mr-1">+</span> Start New Plan
          </button>
          <div>
            <h2 className="text-xs font-medium text-zinc-400 mb-2 uppercase tracking-wider">Active Plan:</h2>
            <div className="flex items-center gap-2">
              <div className={`flex-1 border rounded-md px-3 py-1.5 text-xs truncate flex items-center gap-2 ${cardBg} ${sidebarTxt}`}>
                <span className="w-2 h-2 rounded-full bg-[#29b07a]"></span>
                {activeSession.substring(0,8)}...
              </div>
              <button onClick={(e) => handleDeleteSession(activeSession, e)} className={`border hover:bg-zinc-500/10 rounded-md p-1.5 text-xs text-zinc-400 hover:text-red-400 transition ${isDark ? 'border-[#313543]' : 'border-zinc-300'}`}>🗑️</button>
            </div>
          </div>
          {otherSessions.length > 0 && (
            <div className="space-y-2">
              <h2 className="text-xs font-medium text-zinc-400 mb-2 uppercase tracking-wider">Previous Plans:</h2>
              {otherSessions.map(s => (
                <div key={s} className="flex items-center gap-2">
                  <button 
                    onClick={() => setActiveSession(s)}
                    className={`flex-1 text-left bg-transparent border hover:bg-zinc-500/10 rounded-md px-3 py-1.5 text-xs truncate flex items-center gap-2 transition ${
                      isDark ? 'border-[#313543] text-zinc-300' : 'border-zinc-300 text-zinc-700'
                    }`}
                  >
                    <span>💬</span>{s.substring(0,8)}...
                  </button>
                  <button onClick={(e) => handleDeleteSession(s, e)} className={`bg-transparent border hover:bg-zinc-500/10 rounded-md p-1.5 text-xs text-zinc-400 hover:text-red-400 transition ${isDark ? 'border-[#313543]' : 'border-zinc-300'}`}>🗑️</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Content Viewport Canvas */}
      <div className="print-area flex-1 flex flex-col h-full overflow-hidden relative">
        {!isSidebarOpen && (
          <button onClick={() => setIsSidebarOpen(true)} className="no-print absolute top-5 left-4 z-50 text-zinc-400 hover:text-zinc-200 font-bold text-xs p-1 focus:outline-none transition">»</button>
        )}

        <div className="no-print absolute top-5 right-6 z-50">
          <button onClick={() => setShowStreamlitMenu(!showStreamlitMenu)} className="text-zinc-400 hover:text-zinc-200 font-bold text-xl p-1 focus:outline-none transition">⋮</button>
          {showStreamlitMenu && (
            <div className="absolute right-0 top-7 w-48 bg-[#262730] border border-[#313543] rounded-lg shadow-xl py-1 text-xs text-zinc-200 no-print">
              <div className="px-3 py-2 border-b border-zinc-700 flex justify-between items-center text-[10px] font-bold uppercase text-zinc-400">
                <span>Theme:</span>
                <div className="flex gap-1 bg-[#0e1117] p-0.5 rounded border border-zinc-700">
                  <button onClick={() => { setTheme('light'); setShowStreamlitMenu(false); }} className={`px-1.5 py-0.5 rounded text-[9px] ${!isDark ? 'bg-zinc-200 text-black' : 'text-zinc-400 hover:text-white'}`}>Light</button>
                  <button onClick={() => { setTheme('dark'); setShowStreamlitMenu(false); }} className={`px-1.5 py-0.5 rounded text-[9px] ${isDark ? 'bg-purple-600 text-white' : 'text-zinc-400 hover:text-black'}`}>Dark</button>
                </div>
              </div>
              <button onClick={handleRerun} className="w-full text-left px-4 py-2 hover:bg-zinc-800 flex justify-between items-center transition">
                <span>Rerun</span><span className="text-zinc-500 font-mono text-[10px]">R</span>
              </button>
              <button onClick={() => { setShowStreamlitMenu(false); setTimeout(() => window.print(), 150); }} className="w-full text-left px-4 py-2 hover:bg-zinc-800 flex justify-between items-center border-t border-zinc-700 transition font-medium">
                <span>Print</span>
              </button>
              <button onClick={handleClearAllSessions} className="w-full text-left px-4 py-2 hover:bg-zinc-800 text-red-400 border-t border-zinc-700 flex justify-between items-center transition font-medium">
                <span>Clear global cache</span><span className="text-red-500/50 font-mono text-[10px]">C</span>
              </button>
            </div>
          )}
        </div>

        {/* פריסה ממורכזת נקייה בדיוק כמו בסטרימליט */}
        <div className="print-content-wrapper flex-1 overflow-y-auto px-6 md:px-16 pt-12 pb-32 space-y-8 max-w-5xl mx-auto w-full">
          <h1 className={`print-title text-3xl font-bold tracking-tight flex items-center gap-3 transition-colors ${titleColor}`}>
            <span>✈️</span> Marco — Smart Travel Agent
          </h1>

          <div className="print-kpi grid grid-cols-3 gap-6 pt-2 text-left">
            <div>
              <div className="text-xs text-zinc-400 mb-1 font-medium">Current Thread ID</div>
              <div className={`print-kpi-val text-xl md:text-2xl font-semibold font-mono transition-colors ${isDark ? 'text-white' : 'text-zinc-800'}`}>{activeSession.substring(0,12)}...</div>
            </div>
            <div>
              <div className="text-xs text-zinc-400 mb-1 font-medium">Cache Status</div>
              <div className={`print-kpi-val text-xl md:text-2xl font-semibold transition-colors ${isDark ? 'text-white' : 'text-zinc-800'}`}>{kpiData.cacheStatus}</div>
            </div>
            <div>
              <div className="text-xs text-zinc-400 mb-1 font-medium">Cache Record TTL</div>
              <div className={`print-kpi-val text-xl md:text-2xl font-semibold transition-colors ${isDark ? 'text-white' : 'text-zinc-800'}`}>{kpiData.cacheTtl}</div>
            </div>
          </div>

          <hr className="border-[#313543] opacity-30 my-2 no-print" />

          {criticData && (
            <div className={`no-print border rounded-xl overflow-hidden transition-colors ${cardBg}`}>
              <button onClick={() => setIsCriticExpanded(!isCriticExpanded)} className="w-full px-5 py-3 text-left font-semibold text-xs flex justify-between items-center">
                <span>📊 Critic Evaluation Panel — Quality Score: {criticData.score}/100</span><span>{isCriticExpanded ? '▲' : '▼'}</span>
              </button>
              {isCriticExpanded && (
                <div className={`px-5 pb-5 grid grid-cols-2 gap-6 text-xs border-t pt-4 ${isDark ? 'bg-[#1e1f26] border-zinc-700/50' : 'bg-white border-zinc-200'}`}>
                  <div>
                    <strong className="text-red-400 block mb-2 font-bold">Issues Identified:</strong>
                    {criticData.issues?.length > 0 ? criticData.issues.map((i, idx) => <div key={idx} className="mb-1">⚠️ {i}</div>) : <div>No operational issues identified.</div>}
                  </div>
                  <div>
                    <strong className="text-green-500 block mb-2 font-bold">Optimization Suggestions:</strong>
                    {criticData.suggestions?.length > 0 ? criticData.suggestions.map((s, idx) => <div key={idx} className="mb-1">💡 {s}</div>) : <div>No customization suggestions provided.</div>}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="space-y-6 pt-2">
            {messagesLog.map((msg, idx) => (
              <div key={idx} className="flex flex-col space-y-1">
                <span className="text-[11px] font-bold text-zinc-400 tracking-wider no-print">
                  {msg.role === 'user' ? 'You' : 'Marco (AI Agent)'}
                </span>
                <div className={`print-bubble rounded-lg p-4 text-sm leading-relaxed whitespace-pre-wrap transition-colors ${
                  msg.role === 'user' ? `bg-transparent font-medium ${isDark ? 'text-white' : 'text-zinc-900'}` : `border ${cardBg} ${cardTxt}`
                }`}>
                  <strong className="hidden print:block text-xs uppercase mb-1 opacity-70">
                    {msg.role === 'user' ? 'User Request:' : 'Compiled Itinerary Plan:'}
                  </strong>
                  {msg.content}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className={`no-print border rounded-lg p-4 space-y-2 ${cardBg}`}>
                <div className="text-xs font-bold text-purple-400 flex items-center gap-2">
                  <span className="animate-spin text-sm">⚙️</span> Marco is communicating with sub-agents...
                </div>
                <pre className={`text-xs font-mono p-3 rounded border whitespace-pre-wrap ${isDark ? 'bg-[#0e1117] border-zinc-800 text-zinc-400' : 'bg-white border-zinc-200 text-zinc-600'}`}>
                  {progressLogs}
                </pre>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {hitlPending && (
            <div className={`no-print border border-amber-500/30 rounded-xl p-5 space-y-4 ${cardBg}`}>
              <div className="text-amber-500 font-semibold text-xs flex items-center gap-2">
                <span>✋</span> Human Review Triggered: Please evaluate the compiled travel plan.
              </div>
              <input 
                type="text" 
                placeholder="Enter revision notes (required only when clicking 'Request Edits')..."
                value={feedback}
                disabled={isLoading}
                onChange={e => setFeedback(e.target.value)}
                className={`w-full p-2.5 border rounded-md text-xs focus:outline-none focus:border-amber-500 shadow-inner ${
                  isDark ? 'bg-[#0e1117] border-[#313543] text-white' : 'bg-white border-zinc-300 text-zinc-800'
                }`}
              />
              <div className="flex gap-3 text-xs font-semibold">
                <button onClick={() => handleHitlAction("approved")} disabled={isLoading} className="flex-1 bg-[#29b07a] hover:bg-[#249c6c] disabled:opacity-50 text-white py-2 rounded-md transition shadow-md">👍 Approve Plan</button>
                <button onClick={() => handleHitlAction("edit")} disabled={isLoading} className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white py-2 rounded-md transition shadow-md">📝 Request Edits</button>
                <button onClick={() => handleHitlAction("cancelled")} disabled={isLoading} className="flex-1 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 text-zinc-200 py-2 rounded-md transition">❌ Cancel Plan</button>
              </div>
            </div>
          )}
        </div>

        {!hitlPending && (
          <div className={`no-print absolute bottom-0 left-0 right-0 px-6 md:px-16 pb-8 pt-2 ${isDark ? 'bg-[#0e1117]' : 'bg-[#ffffff]'}`}>
            <div className="max-w-5xl mx-auto w-full">
              <form onSubmit={handleSendMessage} className={`relative border rounded-xl flex items-center px-4 py-2.5 shadow-xl transition-colors ${inputBg}`}>
                <input
                  type="text"
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  disabled={isLoading}
                  placeholder="Where do you want to travel?"
                  className={`flex-1 bg-transparent text-sm focus:outline-none disabled:opacity-50 ${isDark ? 'text-white placeholder-zinc-500' : 'text-zinc-800 placeholder-zinc-400'}`}
                />
                <button type="submit" disabled={isLoading || !prompt.trim()} className={`border text-zinc-300 font-bold p-1.5 rounded-lg disabled:opacity-30 transition flex items-center justify-center w-7 h-7 text-xs ${isDark ? 'bg-[#0e1117] border-[#313543] hover:bg-zinc-800' : 'bg-[#f0f2f6] border-zinc-300 text-zinc-600 hover:bg-zinc-200'}`}>↑</button>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;