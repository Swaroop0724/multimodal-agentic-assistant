"""
ui/app.py — Exact mockup. No Streamlit sidebar. Columns layout.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from pathlib import Path

st.set_page_config(page_title="Multimodal Agentic Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

.stApp { background:#0a0e17 !important; }
html,body,[class*="css"] { font-family:'Syne',sans-serif !important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:0 !important; max-width:100% !important; }
section[data-testid="stSidebar"] { display:none !important; }
[data-testid="collapsedControl"] { display:none !important; }

/* ── LEFT PANEL ── */
.lp { background:#1a1200; border-right:2px solid #f59e0b; min-height:100vh; padding:20px 14px; }
.lp-h { color:#f59e0b; font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin:0 0 10px; }
.lp-row { color:#fbbf24; font-size:1rem; line-height:2.1; display:flex; align-items:center; gap:8px; }
.lp-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.lp-code { color:#fde68a; background:#2d1f00; padding:2px 6px; border-radius:3px; font-family:'JetBrains Mono',monospace; font-size:0.88rem; }
.lp-div { border:none; border-top:1px solid #92400e; margin:10px 0; }
.lp-ref { color:#fbbf24; font-size:0.95rem; line-height:1.7; }
.lp-ref b { color:#f59e0b; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background:#0a0e17 !important; padding:8px 4px 0 !important; gap:4px !important; border-bottom:1px solid #1e293b !important; margin:0 !important; border-radius:0 !important; }
.stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0 !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:1rem !important; color:#cbd5e1 !important; padding:10px 24px !important; background:#1e293b !important; border:1px solid #334155 !important; border-bottom:none !important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#2563eb,#7c3aed) !important; color:#ffffff !important; border-color:#2563eb !important; font-size:1rem !important; }
.stTabs [data-baseweb="tab"]:hover { background:#334155 !important; color:#ffffff !important; }
.stTabs [data-baseweb="tab-panel"] { padding:0 !important; }

/* ── CHAT CONTAINERS ── */
[data-testid=stVerticalBlockBorderWrapper] {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 4px !important;
}
[data-testid=stVerticalBlockBorderWrapper] p { color: #f1f5f9 !important; }
[data-testid=stVerticalBlockBorderWrapper] li { color: #f1f5f9 !important; }

/* ── TEXT ── */
p,li { color:#f1f5f9 !important; }
[data-testid="stMarkdownContainer"] p { color:#f1f5f9 !important; font-size:0.92rem !important; line-height:1.7 !important; }
[data-testid="stMarkdownContainer"] li { color:#f1f5f9 !important; }

/* ── DARK CODE ── */
pre { background:#1e293b !important; border-radius:8px !important; padding:12px !important; border:1px solid #334155 !important; }
pre *,pre span,pre span[style] { color:#e2e8f0 !important; background:transparent !important; }
code { color:#7dd3fc !important; background:#1e293b !important; padding:2px 5px !important; border-radius:3px !important; }
[data-testid="stMarkdownContainer"] pre { background:#1e293b !important; }
[data-testid="stMarkdownContainer"] pre *,[data-testid="stMarkdownContainer"] pre span[style] { color:#e2e8f0 !important; background:transparent !important; }
[data-testid="stMarkdownContainer"] div { background:transparent !important; }
.highlight,.highlight * { background:#1e293b !important; }
.highlight span { color:#e2e8f0 !important; }

/* ── USER MESSAGE ── */
.mu { display:flex; justify-content:flex-end; margin:8px 0; }
.mu-b { background:linear-gradient(135deg,#1d4ed8,#7c3aed); border-radius:14px 14px 4px 14px; padding:10px 16px; font-size:0.9rem; color:#fff !important; max-width:62%; display:inline-block; }

/* ── AGENT MESSAGE ── */
.ma { margin:8px 0; }
.ma-b { background:#111827; border:1px solid #1e293b; border-radius:4px 14px 14px 14px; padding:12px 16px; max-width:80%; }
.ma-b p { color:#f1f5f9 !important; font-size:0.9rem !important; line-height:1.7 !important; margin-bottom:6px !important; }
.ma-b li { color:#f1f5f9 !important; }
.ma-b pre { background:#1e293b !important; border-radius:6px !important; padding:10px !important; border:1px solid #334155 !important; margin:6px 0 !important; }
.ma-b pre *,.ma-b pre span[style] { color:#e2e8f0 !important; background:transparent !important; }
.ma-b code { color:#7dd3fc !important; background:#1e293b !important; }
.tc { display:inline-block; background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3); color:#4ade80 !important; padding:2px 8px; border-radius:10px; font-size:0.7rem !important; font-family:'JetBrains Mono',monospace !important; margin:2px; }
.mm { font-size:0.68rem !important; color:#475569 !important; margin:3px 0 4px; font-family:'JetBrains Mono',monospace !important; }

/* ── BOTTOM INPUT ROW ── */
.stChatInput { background:transparent !important; border:none !important; padding:0 !important; }
[data-testid="stChatInput"] { background:#111827 !important; border:1px solid #334155 !important; border-radius:14px !important; color:#ffffff !important; }
[data-testid="stChatInput"] textarea { background:#111827 !important; color:#ffffff !important; font-size:1rem !important; border:none !important; border-radius:14px !important; min-height:48px !important; caret-color:#ffffff !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#475569 !important; }
[data-testid="stChatInput"] button { background:linear-gradient(135deg,#1d4ed8,#7c3aed) !important; border-radius:10px !important; }

/* ── IMAGE BUTTON ── */
.img-upload-btn .stButton>button { background:rgba(245,158,11,0.15) !important; color:#fbbf24 !important; border:1px solid rgba(245,158,11,0.4) !important; border-radius:10px !important; font-size:0.9rem !important; padding:10px 16px !important; height:52px !important; }

/* ── OTHER BUTTONS ── */
.stButton>button { background:linear-gradient(135deg,#1d4ed8,#7c3aed) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; }

/* ── SELECT / RADIO ── */
.stSelectbox>div>div { background:#111827 !important; border:1px solid #334155 !important; border-radius:8px !important; }
.stSelectbox>div>div * { color:#fff !important; background:#111827 !important; }
.stSelectbox label { color:#94a3b8 !important; font-size:0.8rem !important; }
/* Dropdown popup options */
[data-baseweb="popover"] { background:#111827 !important; }
[data-baseweb="menu"] { background:#111827 !important; border:1px solid #334155 !important; border-radius:8px !important; }
[data-baseweb="menu"] * { background:#111827 !important; color:#ffffff !important; }
[data-baseweb="option"] { background:#111827 !important; color:#ffffff !important; padding:10px 14px !important; }
[data-baseweb="option"]:hover { background:#1e293b !important; color:#ffffff !important; }
[role="option"] { background:#111827 !important; color:#ffffff !important; }
[role="option"]:hover { background:#1e293b !important; }
[role="listbox"] { background:#111827 !important; border:1px solid #334155 !important; border-radius:8px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color:#fff !important; font-size:0.9rem !important; }

/* ── FILE UPLOADER ── */
.stFileUploader { background:rgba(245,158,11,0.06) !important; border:1px solid rgba(245,158,11,0.3) !important; border-radius:8px !important; }
.stFileUploader *,.stFileUploader label { color:#fbbf24 !important; font-size:0.85rem !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; border:1px dashed rgba(245,158,11,0.35) !important; border-radius:6px !important; }
[data-testid="stFileUploaderDropzone"] button { background:rgba(245,158,11,0.15) !important; color:#f59e0b !important; border:1px solid rgba(245,158,11,0.4) !important; border-radius:5px !important; }

/* ── METRIC CARDS ── */
.mc { background:#111827; border:1px solid #1e293b; border-radius:10px; padding:14px; text-align:center; }
.mc-v { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; color:#3b82f6 !important; display:block; line-height:1; margin-bottom:4px; }
.mc-l { font-size:0.65rem !important; color:#64748b !important; text-transform:uppercase; letter-spacing:0.08em; }
.mc-s { font-size:0.65rem !important; color:#475569 !important; margin-top:2px; }
.sb { background:#111827; border-left:3px solid #a855f7; padding:6px 12px; margin:4px 0; border-radius:0 6px 6px 0; font-size:0.78rem !important; color:#c084fc !important; }
</style>
""", unsafe_allow_html=True)

TOOL_ICONS = {"web_search":"🌐","code_executor":"💻","vision_analyzer":"👁️","calculator":"🔢","wikipedia_search":"📚"}
MODEL_OPTIONS = {"Llama 3.3-70B (Best)":"llama-70b","Llama 3.1-8B (Fast)":"llama-8b","Mixtral":"mixtral"}

def plotly_dark():
    return dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8",family="JetBrains Mono"),
                xaxis=dict(gridcolor="#1e293b",linecolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b",linecolor="#1e293b"),
                margin=dict(l=20,r=20,t=40,b=20))

def tcs(calls):
    return "".join(f'<span class="tc">{TOOL_ICONS.get(t,"🔧")} {t}</span>' for t in calls) if calls else ""

def clean(text):
    text = re.sub(r'```\s+python','```python',text)
    text = re.sub(r'```\n((?:def |class |import |from |print|for |if |while ))','```python\n\\1',text)
    return text

# ── LEFT PANEL ───────────────────────────────────────────────────────────────────
def left_panel():
    from config.settings import settings
    g = bool(settings.groq_api_key and settings.groq_api_key!="dummy")
    t = bool(settings.tavily_api_key and settings.tavily_api_key!="dummy")
    e = bool(settings.e2b_api_key and settings.e2b_api_key!="dummy")
    st.markdown(f"""
<div class="lp">
  <div class="lp-h">⚙️ System Status</div>
  <div class="lp-row"><span class="lp-dot" style="background:{'#22c55e' if g else '#ef4444'}"></span>Groq {'Connected' if g else 'Not set'}</div>
  <div class="lp-row"><span class="lp-dot" style="background:{'#22c55e' if t else '#ef4444'}"></span>Tavily {'Connected' if t else 'Not set'}</div>
  <div class="lp-row"><span class="lp-dot" style="background:{'#22c55e' if e else '#ef4444'}"></span>E2B {'Connected' if e else 'Not set'}</div>
  <hr class="lp-div">
  <div class="lp-h">🧰 Tools</div>
  <div class="lp-row">🌐 <span class="lp-code">web_search</span></div>
  <div class="lp-row">💻 <span class="lp-code">code_executor</span></div>
  <div class="lp-row">👁️ <span class="lp-code">vision_analyzer</span></div>
  <div class="lp-row">🔢 <span class="lp-code">calculator</span></div>
  <div class="lp-row">📚 <span class="lp-code">wikipedia</span></div>
  <hr class="lp-div">
  <div class="lp-h">📌 Reference</div>
  <div class="lp-ref"><b>ReAct</b> — reactive loop, best for dynamic tasks.</div><br>
  <div class="lp-ref"><b>Plan-Execute</b> — plan first, best for known steps.</div>
  <hr class="lp-div">
  <div class="lp-ref" style="color:#92400e;font-size:0.78rem;">LangGraph · Groq · Tavily · E2B</div>
</div>""", unsafe_allow_html=True)

# ── TOPBAR ───────────────────────────────────────────────────────────────────────
def topbar(agent_type, model_label):
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
            border-bottom:1px solid #312e81;padding:16px 22px 14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.8rem;
                  color:#a78bfa;margin:0 0 4px;letter-spacing:-0.5px;">
        🤖 Multimodal Agentic Assistant
      </div>
      <div style="font-size:0.75rem;color:#64748b;font-family:'JetBrains Mono',monospace;">
        ReAct vs Plan-Execute &nbsp;·&nbsp; 5 Tools &nbsp;·&nbsp; Groq LLaMA 3.1 &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; Benchmark Harness
      </div>
    </div>
    <div style="background:#111827;border:1px solid #1e293b;border-radius:6px;
                padding:5px 14px;font-size:0.75rem;color:#94a3b8;
                font-family:'JetBrains Mono',monospace;white-space:nowrap;">
      {model_label} ▾
    </div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4);border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#60a5fa;font-family:'JetBrains Mono',monospace;">🔵 ReAct</span>
    <span style="background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.35);border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#fb923c;font-family:'JetBrains Mono',monospace;">🟠 Plan-Execute</span>
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:20px;padding:3px 12px;font-size:0.72rem;color:#fbbf24;font-family:'JetBrains Mono',monospace;">
      Active: {'🔵 ReAct' if agent_type=='ReAct' else '🟠 Plan-Execute'} &nbsp;|&nbsp; {model_label}
    </span>
  </div>
</div>""", unsafe_allow_html=True)

# ── CHAT TAB ─────────────────────────────────────────────────────────────────────
def chat_tab():
    for k,v in [("messages",[]),("agent_obj",None),("cur_model",None),
                ("cur_agent_type",None),("_at","ReAct"),("_ml","Llama 3.3-70B (Best)"),
                ("_img",None),("_show_up",False)]:
        if k not in st.session_state: st.session_state[k]=v

    # Controls row
    c1,c2,c3=st.columns([3,2,1])
    with c1:
        at=st.radio("",["ReAct","Plan-Execute"],horizontal=True,
                    index=0 if st.session_state["_at"]=="ReAct" else 1,
                    label_visibility="collapsed")
        st.session_state["_at"]=at
    with c2:
        ml=st.selectbox("",list(MODEL_OPTIONS.keys()),
                        index=list(MODEL_OPTIONS.keys()).index(st.session_state["_ml"]),
                        label_visibility="collapsed")
        mn=MODEL_OPTIONS[ml]; st.session_state["_ml"]=ml
    with c3:
        if st.button("🗑️ Clear",use_container_width=True):
            st.session_state.messages=[]; st.rerun()

    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f'''
<div style="display:flex;justify-content:flex-end;padding:4px 8px;">
<div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);
            border-radius:18px 18px 4px 18px;
            padding:11px 18px;max-width:60%;
            color:#ffffff;font-size:0.92rem;line-height:1.6;
            font-family:Syne,sans-serif;">
  {msg["content"]}
</div></div>''', unsafe_allow_html=True)
        else:
            st.markdown('''
<div style="display:flex;justify-content:flex-start;padding:4px 8px;">
<div style="background:#111827;border:1px solid #1e293b;
            border-radius:4px 18px 18px 18px;
            padding:14px 18px;max-width:78%;">
''', unsafe_allow_html=True)
            st.markdown(clean(msg["content"]))
            st.markdown('</div></div>', unsafe_allow_html=True)
            if msg.get("tool_calls"):
                st.markdown(f'<div style="padding:0 8px 4px;">{tcs(msg["tool_calls"])}</div>',unsafe_allow_html=True)
            if msg.get("plan"):
                with st.expander("📋 Plan"):
                    for i,s in enumerate(msg["plan"],1):
                        st.markdown(f'<div class="sb">Step {i}: {s}</div>',unsafe_allow_html=True)
            if msg.get("latency_ms"):
                st.markdown(f'<div class="mm" style="padding-left:10px;">⏱ {msg["latency_ms"]:.0f}ms · {msg.get("iterations",0)} iter</div>',unsafe_allow_html=True)

    # ── BOTTOM BAR: Image btn + chat input in one row ──
    st.markdown('<div style="border-top:1px solid #1e293b;background:#080d14;padding:10px 0 4px;">',unsafe_allow_html=True)
    
    img_col, input_col = st.columns([1, 7])
    with img_col:
        st.markdown('<div class="img-upload-btn">',unsafe_allow_html=True)
        if st.button("📷 Image", use_container_width=True):
            st.session_state["_show_up"] = not st.session_state["_show_up"]
        st.markdown('</div>',unsafe_allow_html=True)
    with input_col:
        query = st.chat_input("Ask me anything...")

    st.markdown('</div>',unsafe_allow_html=True)

    # Show uploader if Image button clicked
    if st.session_state["_show_up"]:
        uploaded=st.file_uploader("Upload image",type=["png","jpg","jpeg","webp"],label_visibility="collapsed")
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False,suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read()); st.session_state["_img"]=tmp.name
            st.image(uploaded,width=160)
    else:
        st.session_state["_img"]=None

    if query:
        st.session_state.messages.append({"role":"user","content":query})
        with st.spinner("🔄 Thinking..."):
            try:
                ak="react" if at=="ReAct" else "plan-execute"
                if st.session_state.cur_model!=mn or st.session_state.cur_agent_type!=ak:
                    if ak=="react":
                        from agents.react_agent import ReActAgent
                        st.session_state.agent_obj=ReActAgent(model_name=mn)
                    else:
                        from agents.plan_execute_agent import PlanExecuteAgent
                        st.session_state.agent_obj=PlanExecuteAgent(model_name=mn)
                    st.session_state.cur_model=mn; st.session_state.cur_agent_type=ak
                r=st.session_state.agent_obj.run(query,image_path=st.session_state["_img"])
                st.session_state.messages.append({
                    "role":"assistant","content":r.output,"tool_calls":r.tool_calls,
                    "latency_ms":r.latency_ms,"iterations":r.iterations,"plan":r.plan,
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

# ── BENCHMARK TAB ────────────────────────────────────────────────────────────────
def benchmark_tab():
    st.markdown("### 📊 Benchmark Runner")
    c1,c2,c3=st.columns(3)
    with c1: qc=st.slider("Queries",5,50,10,5)
    with c2:
        ml=st.selectbox("Model",list(MODEL_OPTIONS.keys()),key="bm"); mn=MODEL_OPTIONS[ml]
    with c3:
        st.markdown("<br>",unsafe_allow_html=True)
        run=st.button("🚀 Run",use_container_width=True)
    if run:
        pb=st.progress(0); sb=st.empty()
        try:
            from benchmark.harness import BenchmarkHarness
            from agents.react_agent import ReActAgent
            from agents.plan_execute_agent import PlanExecuteAgent
            from benchmark.metrics import compare_agents
            h=BenchmarkHarness(query_count=qc,models=[mn]); qs=h._load_queries()
            total=len(qs)*2; done=0; all_m=[]
            ra=ReActAgent(model_name=mn); pe=PlanExecuteAgent(model_name=mn)
            for ag,at in [(ra,"react"),(pe,"plan_execute")]:
                for q in qs:
                    sb.info(f"{'🔵' if at=='react' else '🟠'} {done%len(qs)+1}/{len(qs)}")
                    all_m.append(h._run_single_query(ag,at,mn,q)); done+=1
                    pb.progress(done/total); time.sleep(0.3)
            rm=[m for m in all_m if m.agent_type=="react"]; pm=[m for m in all_m if m.agent_type=="plan_execute"]
            comp=compare_agents(rm,pm); r=comp["react"]; p=comp["plan_execute"]
            sb.success("✅ Done!")
            cols=st.columns(4)
            for col,(lbl,val,sub) in zip(cols,[
                ("ROUGE-L",f"{r['avg_rouge_l']:.4f}",f"vs {p['avg_rouge_l']:.4f}"),
                ("Tool Acc",f"{r['avg_tool_accuracy']*100:.1f}%",f"vs {p['avg_tool_accuracy']*100:.1f}%"),
                ("Latency",f"{r['avg_latency_ms']:.0f}ms",f"vs {p['avg_latency_ms']:.0f}ms"),
                ("Success",f"{r['success_rate']*100:.0f}%",f"{p['success_rate']*100:.0f}%"),
            ]):
                with col: st.markdown(f'<div class="mc"><span class="mc-v">{val}</span><div class="mc-l">{lbl}</div><div class="mc-s">{sub}</div></div>',unsafe_allow_html=True)
            rows=[{"ID":m.query_id,"Cat":m.category,"Agent":m.agent_type,"ROUGE-L":f"{m.rouge_l:.3f}",
                   "Acc":f"{m.tool_accuracy:.2f}","ms":f"{m.latency_ms:.0f}",
                   "Tools":",".join(m.tool_calls_made) or "—","✓":"✅" if m.success else "❌"} for m in all_m]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        except Exception as e: st.error(f"Error: {e}")
    st.divider()
    rd=Path("data/results")
    fs=[f for f in sorted(rd.glob("benchmark_*.csv")) if "summary" not in f.name] if rd.exists() else []
    if fs:
        sel=st.selectbox("Load",[f.name for f in fs],index=len(fs)-1)
        if sel: st.dataframe(pd.read_csv(rd/sel),use_container_width=True,hide_index=True)
    else: st.info("No results yet.")

# ── ANALYTICS TAB ────────────────────────────────────────────────────────────────
def analytics_tab():
    st.markdown("### 📈 Analytics")
    rd=Path("data/results")
    fs=[f for f in sorted(rd.glob("benchmark_*.csv")) if "summary" not in f.name] if rd.exists() else []
    if not fs:
        st.info("Run a benchmark first.")
        fig=go.Figure(data=[go.Bar(name="ReAct",x=["ROUGE-L","Tool Acc","Success"],y=[0.71,87.3,94.0],marker_color="#3b82f6"),
                             go.Bar(name="Plan-Execute",x=["ROUGE-L","Tool Acc","Success"],y=[0.63,71.2,88.0],marker_color="#f97316")])
        fig.update_layout(barmode="group",legend=dict(bgcolor="rgba(0,0,0,0)"),**plotly_dark())
        st.plotly_chart(fig,use_container_width=True); return
    sel=st.selectbox("Run",[f.name for f in fs],index=len(fs)-1)
    df=pd.read_csv(rd/sel); rdf=df[df["agent_type"]=="react"]; pdf=df[df["agent_type"]=="plan_execute"]
    c1,c2,c3=st.columns(3)
    for col,title,yr,yp in [(c1,"ROUGE-L",[rdf["rouge_l"].mean()],[pdf["rouge_l"].mean()]),
                             (c2,"Tool Acc %",[rdf["tool_accuracy"].mean()*100],[pdf["tool_accuracy"].mean()*100]),
                             (c3,"Latency ms",[rdf["latency_ms"].mean()],[pdf["latency_ms"].mean()])]:
        with col:
            f=go.Figure(data=[go.Bar(name="ReAct",x=[title],y=yr,marker_color="#3b82f6",width=0.4),
                               go.Bar(name="Plan-Execute",x=[title],y=yp,marker_color="#f97316",width=0.4)])
            f.update_layout(title=title,barmode="group",showlegend=(col is c1),legend=dict(bgcolor="rgba(0,0,0,0)"),**plotly_dark())
            st.plotly_chart(f,use_container_width=True)
    st.divider(); st.markdown("#### 🎯 Resume Bullet")
    ra=rdf["tool_accuracy"].mean()*100; pa=pdf["tool_accuracy"].mean()*100
    rl=rdf["latency_ms"].mean(); pl=pdf["latency_ms"].mean()
    adv=(pl-rl)/pl*100 if pl>0 else 0
    st.code(f"Benchmarked ReAct vs Plan-Execute — ReAct: {ra:.1f}% vs {pa:.1f}% accuracy, {adv:.0f}% lower latency. LangGraph+Groq+Tavily+E2B.",language=None)

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    left, main_col = st.columns([1, 5])
    with left:
        left_panel()
    with main_col:
        # Topbar FIRST
        at = st.session_state.get("_at","ReAct")
        ml = st.session_state.get("_ml","Llama 3.3-70B (Best)")
        topbar(at, ml)
        # Tabs BELOW topbar
        tab1,tab2,tab3=st.tabs(["💬 Chat","📊 Benchmark","📈 Analytics"])
        with tab1: chat_tab()
        with tab2: benchmark_tab()
        with tab3: analytics_tab()

if __name__=="__main__":
    main()