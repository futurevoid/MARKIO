import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import io, wave, time
from audio_watermarking_ga import (
    PRESETS as GA_PRESETS,
    GeneticAlgorithm,
    generate_watermark,
    prepare_audio_source,
    embed_watermark,
    extract_watermark,
    compute_objectives,
)

st.set_page_config(
    page_title="MARKIO | Audio Watermark Optimization",
    page_icon="🧬", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  :root {
    --markio-bg:#10131d;
    --markio-panel:#191d2b;
    --markio-panel-2:#202536;
    --markio-line:#2b3146;
    --markio-text:#e5e9f7;
    --markio-muted:#8d96b3;
    --markio-cyan:#00d4ff;
    --markio-amber:#ffcc44;
    --markio-green:#6dffa0;
    --markio-coral:#ff6b6b;
  }

  /* ── Layout ─────────────────────────────────────────────────── */
  .block-container { padding-top:3.2rem; padding-bottom:.6rem; }
  div[data-testid="stSidebarContent"] { background:var(--markio-bg); }

  /* ── MARKIO identity ────────────────────────────────────────── */
  .brand-lockup {
    display:flex; align-items:center; gap:12px; margin:.15rem 0 1rem 0;
  }
  .brand-mark {
    width:40px; height:40px; border:1px solid #00d4ff55; border-radius:8px;
    display:grid; place-items:center; color:var(--markio-cyan);
    font-size:1.45rem; font-weight:900; letter-spacing:-.04em;
    background:linear-gradient(145deg,#172033,#0d1019);
    box-shadow:0 0 24px rgba(0,212,255,.08), inset 0 0 16px rgba(0,212,255,.05);
  }
  .brand-name {
    color:var(--markio-text); font-size:1.15rem; font-weight:900;
    letter-spacing:.14em; line-height:1;
  }
  .brand-tag {
    color:var(--markio-muted); font-size:.72rem; letter-spacing:.04em;
    margin-top:5px;
  }

  /* ── Product hero ────────────────────────────────────────────── */
  .hero-wrap {
    border:1px solid var(--markio-line); border-radius:8px;
    padding:22px 24px 20px 24px; margin-bottom:1rem;
    background:
      linear-gradient(90deg,rgba(0,212,255,.08),transparent 46%),
      linear-gradient(180deg,#1a1f2f,#141824);
  }
  .hero-kicker {
    color:var(--markio-cyan); font-size:.68rem; letter-spacing:.18em;
    text-transform:uppercase; font-weight:800; margin-bottom:8px;
  }
  .hero-title {
    color:var(--markio-text); font-size:2.45rem; font-weight:900;
    letter-spacing:.06em; line-height:1; margin-bottom:.55rem;
  }
  .hero-sub { color:var(--markio-muted); font-size:.95rem; margin-bottom:0; max-width:920px; line-height:1.65; }
  .hero-sub code {
    color:var(--markio-text); background:#0c101a; border:1px solid #283044;
    border-radius:6px; padding:1px 6px;
  }

  /* ── Section headers ─────────────────────────────────────────── */
  .sec {
    display:flex; align-items:center; gap:8px;
    font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;
    color:var(--markio-muted); margin:1.4rem 0 .5rem 0; font-weight:700;
  }
  .sec::after {
    content:""; flex:1; height:1px;
    background:linear-gradient(90deg,var(--markio-line),transparent);
  }

  /* ── Metric cards ────────────────────────────────────────────── */
  .metric-card {
    background:var(--markio-panel); border:1px solid var(--markio-line);
    border-radius:8px; padding:16px 14px; text-align:center;
    transition:border-color .2s, transform .2s, box-shadow .2s;
    position:relative; overflow:hidden;
  }
  .metric-card:hover {
    border-color:#00d4ff44; transform:translateY(-2px);
    box-shadow:0 6px 24px rgba(0,212,255,.08);
  }
  .metric-card::before {
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--accent, var(--markio-cyan)); opacity:.78;
  }
  .metric-card .lbl {
    color:var(--markio-muted); font-size:.68rem; letter-spacing:.1em;
    text-transform:uppercase; margin-bottom:6px;
  }
  .metric-card .val {
    color:var(--accent,var(--markio-cyan)); font-size:1.55rem;
    font-weight:800; font-family:monospace; line-height:1.1;
  }
  .metric-card .sub { font-size:.73rem; margin-top:5px; color:var(--markio-muted); }
  .pos { color:var(--markio-green); } .neg { color:var(--markio-coral); } .neu { color:var(--markio-muted); }

  /* ── Gene chromosome ─────────────────────────────────────────── */
  .gene-wrap {
    background:var(--markio-panel); border:1px solid var(--markio-line);
    border-radius:8px; padding:16px 20px; margin-top:10px;
  }
  .gene-header {
    display:flex; justify-content:space-between; align-items:center;
    margin-bottom:12px;
  }
  .gene-lbl { color:var(--markio-muted); font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; }
  .gene-score { color:var(--markio-amber); font-size:.75rem; font-family:monospace; }
  .gene-row { display:flex; align-items:center; gap:10px; margin:7px 0; }
  .gene-name { color:var(--markio-text); font-size:.8rem; width:58px; font-family:monospace; }
  .gene-track {
    flex:1; background:#0e1019; border-radius:6px; height:13px;
    overflow:hidden; border:1px solid #1e2235;
  }
  .gene-fill { height:100%; border-radius:6px; position:relative; }
  .gene-fill::after {
    content:""; position:absolute; top:0; left:0; right:0; bottom:0;
    background:linear-gradient(90deg,transparent 60%,rgba(255,255,255,.15));
  }
  .gene-val { color:var(--markio-text); font-size:.78rem; width:100px; text-align:right; font-family:monospace; }

  /* ── Log box ─────────────────────────────────────────────────── */
  .log-box {
    background:#0a0c14; border:1px solid var(--markio-line);
    border-radius:8px; padding:12px 14px; font-family:"JetBrains Mono",monospace;
    font-size:.73rem; color:#b0b8d0; max-height:240px;
    overflow-y:auto; line-height:1.7;
    scrollbar-width:thin; scrollbar-color:#2a2d40 transparent;
  }
  .log-box::-webkit-scrollbar { width:4px; }
  .log-box::-webkit-scrollbar-track { background:transparent; }
  .log-box::-webkit-scrollbar-thumb { background:#2a2d40; border-radius:4px; }

  /* ── Badges ──────────────────────────────────────────────────── */
  .badge {
    display:inline-block; padding:2px 8px; border-radius:20px;
    font-size:.67rem; font-weight:700; letter-spacing:.04em;
  }
  .b-g   { background:#0d2014; color:#6dffa0; border:1px solid #1d5a2a; }
  .b-inj { background:#2d1f4a; color:#b57bee; border:1px solid #5a3a8a; }

  /* ── Live metrics strip ──────────────────────────────────────── */
  .live-strip {
    display:flex; gap:16px; background:var(--markio-panel);
    border:1px solid var(--markio-line); border-radius:8px;
    padding:10px 18px; margin:10px 0; align-items:center;
  }
  .live-lbl { color:var(--markio-muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
  .live-val { color:var(--markio-cyan); font-size:1.15rem; font-weight:700; font-family:monospace; }
  .live-sep { color:#2a2d40; font-size:1.2rem; }

  /* ── Audio player card ───────────────────────────────────────── */
  .audio-card {
    background:var(--markio-panel); border:1px solid var(--markio-line); border-radius:8px;
    padding:14px 16px; margin-bottom:4px;
  }
  .audio-card-lbl {
    font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
    color:var(--markio-muted); margin-bottom:8px; display:flex; align-items:center; gap:6px;
  }
  .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

  /* ── Preset buttons ──────────────────────────────────────────── */
  .preset-row { display:flex; gap:6px; margin-bottom:8px; }

  /* ── Feature cards (empty state) ────────────────────────────── */
  .feat-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:20px; }
  .feat-card {
    background:var(--markio-panel); border:1px solid var(--markio-line); border-radius:8px; padding:16px;
  }
  .feat-icon { color:var(--markio-cyan); font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; font-weight:800; margin-bottom:8px; }
  .feat-title { color:var(--markio-text); font-size:.85rem; font-weight:700; margin-bottom:4px; }
  .feat-desc { color:var(--markio-muted); font-size:.76rem; line-height:1.5; }

  /* ── Tab styling ─────────────────────────────────────────────── */
  button[data-baseweb="tab"] {
    font-size:.8rem !important; padding:6px 14px !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color:var(--markio-cyan) !important; border-bottom-color:var(--markio-cyan) !important;
  }
  @media (max-width: 760px) {
    .hero-title { font-size:2rem; }
    .feat-grid { grid-template-columns:1fr; }
    .live-strip { flex-wrap:wrap; }
  }
</style>
""", unsafe_allow_html=True)




BG    = "#12141f"; PANEL = "#1c1f30"
CYAN  = "#00d4ff"; CORAL = "#ff6b6b"
GREEN = "#6dffa0"; AMBER = "#ffcc44"
PURP  = "#b57bee"; GREY  = "#8890aa"

def _base(fig, title="", h=300):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#dde1f0", size=12)),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=GREY, size=10), height=h,
        margin=dict(l=48, r=16, t=38, b=38),
        legend=dict(bgcolor=PANEL, bordercolor="#2a2d40", borderwidth=1, font=dict(size=9)),
        xaxis=dict(gridcolor="#22263a", zerolinecolor="#22263a"),
        yaxis=dict(gridcolor="#22263a", zerolinecolor="#22263a"))
    return fig

def chart_waves(audio, wm_audio, sr):
    t   = np.linspace(0, len(audio)/sr, len(audio))[:2000]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Original", "Watermarked", "Noise (difference)"])
    for col, (y, c) in enumerate(
        [(audio[:2000], CYAN), (wm_audio[:2000], CORAL), ((wm_audio-audio)[:2000], GREEN)], 1):
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines",
                                 line=dict(color=c, width=0.8), showlegend=False), row=1, col=col)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=210,
                      margin=dict(l=32, r=8, t=30, b=30), font=dict(color=GREY, size=9))
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{ax: dict(gridcolor="#22263a", zerolinecolor="#22263a")})
    fig.update_annotations(font_color="#dde1f0", font_size=10)
    return fig

def chart_convergence(hist):
    g   = list(range(1, len(hist["hof_snr"])+1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=g, y=hist["hof_snr"], name="SNR (dB)",
                             line=dict(color=CORAL, width=1.8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=g, y=[a*100 for a in hist["hof_acc"]],
                             name="Bit Accuracy (%)",
                             line=dict(color=GREEN, width=1.8, dash="dash")), secondary_y=True)
    for rg in hist["restarts"]:
        fig.add_vline(x=rg+1, line=dict(color=PURP, width=1, dash="dot"))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=300,
                      margin=dict(l=50, r=52, t=38, b=38),
                      title=dict(text="HoF Objectives per Generation",
                                 font=dict(color="#dde1f0", size=12)),
                      legend=dict(bgcolor=PANEL, bordercolor="#2a2d40"),
                      xaxis=dict(gridcolor="#22263a", title="Generation"),
                      yaxis=dict(gridcolor="#22263a", color=CORAL, title="SNR (dB)"),
                      yaxis2=dict(gridcolor="#22263a", color=GREEN, title="Accuracy (%)"))
    return fig

def chart_front_size(hist):
    g   = list(range(1, len(hist["front0_size"])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=hist["front0_size"], name="Pareto front size",
                             line=dict(color=AMBER, width=1.6),
                             fill="tozeroy", fillcolor="rgba(255,204,68,0.07)"))
    fig.add_trace(go.Scatter(x=g, y=[d*100 for d in hist["div"]],
                             name="Diversity x100",
                             line=dict(color=PURP, width=1.4, dash="dash")))
    for rg in hist["restarts"]:
        fig.add_vline(x=rg+1, line=dict(color=PURP, width=1, dash="dot"),
                      annotation_text="inject", annotation_font_color=PURP,
                      annotation_font_size=8)
    return _base(fig, "Pareto Front Size & Diversity")

def chart_pareto(final_pop, pareto_front, hof):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p.snr for p in final_pop],
                             y=[p.acc*100 for p in final_pop],
                             mode="markers", name="All individuals",
                             marker=dict(color=GREY, size=5, opacity=0.35)))
    pf = sorted(pareto_front, key=lambda p: p.snr)
    fig.add_trace(go.Scatter(x=[p.snr for p in pf], y=[p.acc*100 for p in pf],
                             mode="markers+lines", name="Pareto front (rank 0)",
                             marker=dict(color=AMBER, size=8),
                             line=dict(color=AMBER, width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=[hof.snr], y=[hof.acc*100], mode="markers",
                             name="HoF (best balanced)",
                             marker=dict(color=CYAN, size=14, symbol="star")))
    return _base(fig, "Final Pareto Front — SNR vs Robustness  (no scalar weighting)", h=320)

def chart_sweep(audio, wm, hof, gene="alpha"):
    if gene == "alpha":
        sweep = np.linspace(0.001, 0.15, 50)
        snrs, accs = [], []
        for a in sweep:
            s, c = compute_objectives(a, hof.band, hof.n_carriers, audio, wm)
            snrs.append(s); accs.append(c*100)
        vline  = hof.alpha
        label  = f"HoF alpha={hof.alpha:.4f}"
        title  = f"alpha Sweep  (band={hof.band:.2f}, nc={hof.n_carriers})"
        xlabel = "alpha"
    else:
        sweep = np.linspace(0.0, 1.0, 30)
        snrs, accs = [], []
        for b in sweep:
            s, c = compute_objectives(hof.alpha, b, hof.n_carriers, audio, wm)
            snrs.append(s); accs.append(c*100)
        vline  = hof.band
        label  = f"HoF band={hof.band:.2f}"
        title  = f"Band Sweep  (alpha={hof.alpha:.4f}, nc={hof.n_carriers})"
        xlabel = "band (0=narrow, 1=full)"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sweep, y=snrs, name="SNR (dB)",
                             line=dict(color=CORAL, width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sweep, y=accs, name="Accuracy (%)",
                             line=dict(color=GREEN, width=1.5, dash="dash")), secondary_y=True)
    fig.add_vline(x=vline, line=dict(color=CYAN, width=1.3, dash="dash"),
                  annotation_text=label, annotation_font_color=CYAN, annotation_font_size=9)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=270,
                      margin=dict(l=48, r=52, t=38, b=38),
                      title=dict(text=title, font=dict(color="#dde1f0", size=12)),
                      legend=dict(bgcolor=PANEL, bordercolor="#2a2d40"),
                      xaxis=dict(gridcolor="#22263a", title=xlabel),
                      yaxis=dict(gridcolor="#22263a", color=CORAL, title="SNR (dB)"),
                      yaxis2=dict(gridcolor="#22263a", color=GREEN, title="Accuracy (%)"))
    return fig

def chart_bits(audio, wm, hof):
    wma = embed_watermark(audio, wm, hof.alpha, hof.band, hof.n_carriers)
    ext = extract_watermark(audio, wma, len(wm), hof.band, hof.n_carriers)
    acc = np.mean(ext == wm) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(wm))), y=wm, name="Original",
                         marker_color=CYAN, opacity=0.55, width=0.8))
    fig.add_trace(go.Bar(x=list(range(len(wm))), y=ext, name="Extracted",
                         marker_color=CORAL, opacity=0.9, width=0.4))
    return _base(fig, f"Bit Recovery (clean signal) — Acc: {acc:.1f}%", h=260)


def to_wav(sig, sr=8000):
    buf = io.BytesIO()
    pcm = np.int16(np.clip(sig, -1, 1) * 32767)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def mcard(label, value, delta=None, pos=True, accent="#00d4ff"):
    d = ""
    if delta is not None:
        cls   = "pos" if pos else "neg"
        arrow = "▲" if pos else "▼"
        d     = f'<div class="sub {cls}">{arrow} {delta}</div>'
    return (f'<div class="metric-card" style="--accent:{accent}">'
            f'<div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>{d}</div>')

def gene_html(hof):
    norms  = [(hof.alpha-0.001)/(0.15-0.001), hof.band, (hof.genes[2]-1.0)/2.0]
    names  = ["alpha", "band", "nc"]
    vals   = [f"{hof.alpha:.5f}", f"{hof.band:.3f}", f"{hof.n_carriers}  (int)"]
    colors = [CYAN, AMBER, GREEN]
    score  = hof.balanced_score()
    rows   = "".join(
        f'<div class="gene-row">'
        f'<div class="gene-name">{n}</div>'
        f'<div class="gene-track"><div class="gene-fill" style="width:{v*100:.1f}%;background:{c}"></div></div>'
        f'<div class="gene-val">{lv}</div></div>'
        for n, v, c, lv in zip(names, norms, colors, vals))
    return (f'<div class="gene-wrap">'
            f'<div class="gene-header">'
            f'<div class="gene-lbl">MARKIO HoF Chromosome</div>'
            f'<div class="gene-score">balanced score {score:.4f}</div>'
            f'</div>{rows}</div>')


PRESET_LABELS = {
    "Quick": "quick",
    "Balanced": "balanced",
    "Complete": "complete",
}

if "preset" not in st.session_state or st.session_state.preset not in PRESET_LABELS:
    st.session_state.preset = "Balanced"

with st.sidebar:
    st.markdown("""
    <div class="brand-lockup">
      <div class="brand-mark">🧬</div>
      <div>
        <div class="brand-name">MARKIO</div>
        <div class="brand-tag">Audio watermark optimization</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Presets ──────────────────────────────────────────────────
    st.markdown('<div class="sec">Preset</div>', unsafe_allow_html=True)
    chosen_preset = st.radio("Preset", list(PRESET_LABELS.keys()),
                             index=list(PRESET_LABELS.keys()).index(st.session_state.preset),
                             horizontal=True, label_visibility="collapsed")
    st.session_state.preset = chosen_preset
    P = GA_PRESETS[PRESET_LABELS[chosen_preset]]

    # ── Audio ────────────────────────────────────────────────────
    st.markdown('<div class="sec">Audio</div>', unsafe_allow_html=True)
    audio_src = st.radio("Audio source", ["Generate audio", "Upload audio (WAV)"],
                         horizontal=True, label_visibility="collapsed")
    uploaded_wav = None
    if audio_src == "Upload audio (WAV)":
        uploaded_wav = st.file_uploader(
            "Audio file (WAV, AIFF, FLAC)", type=["wav", "aif", "aiff", "flac"],
            help="Any sample rate — will be resampled to the rate selected below.")
        if uploaded_wav is not None:
            st.success(f"Loaded {uploaded_wav.name}")

    sr = st.selectbox("Sample rate (Hz)", [8000, 16000], index=0)
    if audio_src == "Generate audio":
        duration = st.slider("Duration (s)", 1.0, 4.0, 2.0, 0.5)
    else:
        duration = None
    wm_bits = st.selectbox("Watermark bits", [32, 64, 128], index=1)


    # ── GA Parameters ────────────────────────────────────────────
    st.markdown('<div class="sec">GA Parameters</div>', unsafe_allow_html=True)
    pop_size = st.slider("Population size",      10,  80,  P["pop_size"], 5)
    n_gens   = st.slider("Generations",          20,  120, P["generations"],  10)

    with st.expander("Advanced"):
        sbx_eta  = st.slider("SBX eta",               1,   20,  P["sbx_eta"])
        k_min    = st.slider("Tournament k min",       2,   4,   P["k_min"])
        k_max    = st.slider("Tournament k max",       3,   10,  P["k_max"])
        patience = st.slider("Stagnation patience",    5,   30,  P["patience"])
        inj_frac = st.slider("Injection fraction",     0.1, 0.5, P["inj_frac"], 0.05)
        seed     = st.number_input("Random seed",      0,   999, 0, 1)

    st.markdown("")
    run_btn = st.button(
        "Run MARKIO Optimization",
        type="primary",
        use_container_width=True,
    )
    st.divider()
    st.markdown("""<div style="color:#8890aa;font-size:.74rem;line-height:1.7">
    <b style="color:#dde1f0">MARKIO chromosome</b><br>
    <span style="color:#00d4ff">alpha</span> · embedding strength<br>
    <span style="color:#ffcc44">band</span> · carrier bandwidth<br>
    <span style="color:#6dffa0">nc</span> · carriers per bit<br><br>
    <b style="color:#dde1f0">Optimization engine</b><br>
    Non-dominated sorting + crowding distance.<br>
    No weighted sum — SNR and accuracy treated independently.
    </div>""", unsafe_allow_html=True)


st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-kicker">Secure signal lab</div>'
    '<div class="hero-title">MARKIO</div>'
    '<div class="hero-sub">A multi-objective audio watermarking workspace that evolves '
    '<b style="color:#e5e9f7">[alpha · band · n_carriers]</b> to balance '
    '<b style="color:#ff6b6b">signal fidelity</b> and '
    '<b style="color:#6dffa0">bit recovery</b>. MARKIO builds the Pareto front directly, '
    'so robustness and imperceptibility stay visible as independent objectives.</div>'
    '</div>',
    unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    st.session_state.results = None

    # ── Audio source ──────────────────────────────────────────────
    source_mode = "generate" if audio_src == "Generate audio" else "wav"
    if source_mode == "wav":
        if uploaded_wav is None:
            st.error("Please upload an audio file before running the GA.")
            st.stop()
        with st.spinner("Loading audio…"):
            audio, sample_rate, source_label = prepare_audio_source(
                source_mode, uploaded_wav, sr, duration)
        st.info(f"MARKIO source loaded: **{uploaded_wav.name}** — "
                f"{len(audio)/sample_rate:.2f}s · {sample_rate} Hz · "
                f"{len(audio):,} samples")
    else:
        audio, sample_rate, source_label = prepare_audio_source(
            source_mode, None, sr, duration)
    # ─────────────────────────────────────────────────────────────
    watermark = generate_watermark(length=wm_bits)
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        generations=n_gens,
        sbx_eta=sbx_eta,
        crossover_prob=0.85,
        tournament_k_min=k_min,
        tournament_k_max=k_max,
        stagnation_patience=patience,
        injection_frac=inj_frac,
        seed=int(seed),
        verbose=False,
    )
    b_snr, b_acc = compute_objectives(0.05, 1.0, 1, audio, watermark)

    prog      = st.progress(0, text="Initializing MARKIO population...")
    live_slot = st.empty()
    log_slot  = st.empty()
    log_rows  = []
    t0        = time.time()
    hof = pareto_front = final_pop = hist = None

    for result in ga.iter_run(audio, watermark):
        if result[3] == "__done__":
            _, hof, hist, _, final_pop, pareto_front = result
            break
        gen, hof, hist, note = result[0], result[1], result[2], result[3]
        nb  = f' <span class="badge b-inj">{note}</span>' if note else ""
        row = (f'<span class="badge b-g">Gen {gen:>3}</span>  '
               f'alpha={hof.alpha:.4f}  band={hof.band:.2f}  nc={hof.n_carriers}  '
               f'SNR={hof.snr:.1f}dB  Acc={hof.acc*100:.1f}%{nb}')
        log_rows.append(row)
        if len(log_rows) > 50: log_rows.pop(0)
        pct = gen / n_gens
        prog.progress(pct, text=f"MARKIO generation {gen} / {n_gens}  ·  {pct*100:.0f}%")
        live_slot.markdown(
            f'<div class="live-strip">'
            f'<div><div class="live-lbl">SNR</div><div class="live-val">{hof.snr:.2f} dB</div></div>'
            f'<div class="live-sep">|</div>'
            f'<div><div class="live-lbl">Bit Accuracy</div><div class="live-val" style="color:#6dffa0">{hof.acc*100:.1f}%</div></div>'
            f'<div class="live-sep">|</div>'
            f'<div><div class="live-lbl">Alpha</div><div class="live-val" style="color:#ffcc44">{hof.alpha:.5f}</div></div>'
            f'<div class="live-sep">|</div>'
            f'<div><div class="live-lbl">Pareto front</div><div class="live-val" style="color:#b57bee">{hist["front0_size"][-1]} gen</div></div>'
            f'</div>', unsafe_allow_html=True)
        log_slot.markdown('<div class="log-box">' + "<br>".join(log_rows) + "</div>",
                          unsafe_allow_html=True)

    elapsed = time.time() - t0
    prog.progress(1.0, text=f"MARKIO complete in {elapsed:.1f}s — Pareto front: {len(pareto_front)} solutions")
    live_slot.empty()
    st.session_state.results = dict(
        audio=audio, sr=sample_rate, watermark=watermark,
        hof=hof, hist=hist, pareto_front=pareto_front,
        final_pop=final_pop, b_snr=b_snr, b_acc=b_acc,
        audio_src=audio_src,
        audio_name=source_label)
    st.rerun()

if st.session_state.results:
    R       = st.session_state.results
    hof     = R["hof"]
    h       = R["hist"]
    audio   = R["audio"]
    sr_val  = R["sr"]
    wm      = R["watermark"]
    wm_opt  = embed_watermark(audio, wm, hof.alpha, hof.band, hof.n_carriers)
    d_snr   = hof.snr - R["b_snr"]
    d_acc   = (hof.acc - R["b_acc"]) * 100
    noise   = wm_opt - audio

    # ── Source banner ─────────────────────────────────────────────
    if R.get("audio_name"):
        st.info(f"MARKIO source: **{R['audio_name']}**  ·  "
                f"{len(audio)/sr_val:.2f}s  ·  {sr_val} Hz  ·  {len(audio):,} samples")

    # ── Metric cards ──────────────────────────────────────────────
    st.markdown('<div class="sec">MARKIO Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(mcard("SNR", f"{hof.snr:.1f} dB",
        f"{d_snr:+.1f} dB vs baseline", d_snr >= 0, "#ff6b6b"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("Bit Accuracy", f"{hof.acc*100:.1f}%",
        f"{d_acc:+.1f}% vs baseline", d_acc >= 0, "#6dffa0"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("Alpha", f"{hof.alpha:.5f}",
        "embedding strength", True, "#00d4ff"), unsafe_allow_html=True)
    with c4: st.markdown(mcard("Band", f"{hof.band:.3f}",
        "0=narrow  1=full", True, "#ffcc44"), unsafe_allow_html=True)
    with c5: st.markdown(mcard("n_carriers", str(hof.n_carriers),
        "per bit", True, "#b57bee"), unsafe_allow_html=True)
    with c6: st.markdown(mcard("Pareto front", str(len(R["pareto_front"])),
        "solutions", True, "#8890aa"), unsafe_allow_html=True)

    st.markdown(gene_html(hof), unsafe_allow_html=True)
    st.markdown("<div style='margin:.6rem 0'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "Waveforms",
        "Convergence",
        "Pareto Front",
        "Gene Sweeps",
        "Bit Recovery",
    ])

    with t1:
        st.plotly_chart(chart_waves(audio, wm_opt, sr_val), use_container_width=True)
        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown(
                '<div class="audio-card"><div class="audio-card-lbl">'
                '<span class="dot" style="background:#00d4ff"></span> Original</div></div>',
                unsafe_allow_html=True)
            st.audio(to_wav(audio, sr_val), format="audio/wav")
        with cb:
            st.markdown(
                '<div class="audio-card"><div class="audio-card-lbl">'
                '<span class="dot" style="background:#ff6b6b"></span> Watermarked</div></div>',
                unsafe_allow_html=True)
            st.audio(to_wav(wm_opt, sr_val), format="audio/wav")
        with cc:
            st.markdown(
                '<div class="audio-card"><div class="audio-card-lbl">'
                '<span class="dot" style="background:#6dffa0"></span> '
                f'Noise signal  <span style="color:#ffcc44;font-family:monospace">'
                f'RMS {np.sqrt(np.mean(noise**2))*100:.3f}%</span></div></div>',
                unsafe_allow_html=True)
            st.audio(to_wav(noise * 10, sr_val), format="audio/wav")

        # Download button
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download MARKIO watermarked audio",
            data=to_wav(wm_opt, sr_val),
            file_name="markio_watermarked.wav",
            mime="audio/wav",
            use_container_width=False,
        )

    with t2:
        col1, col2 = st.columns([3, 2])
        with col1: st.plotly_chart(chart_convergence(h), use_container_width=True)
        with col2: st.plotly_chart(chart_front_size(h), use_container_width=True)

    with t3:
        st.plotly_chart(chart_pareto(R["final_pop"], R["pareto_front"], hof),
                        use_container_width=True)

    with t4:
        col3, col4 = st.columns(2)
        with col3: st.plotly_chart(chart_sweep(audio, wm, hof, "alpha"), use_container_width=True)
        with col4: st.plotly_chart(chart_sweep(audio, wm, hof, "band"),  use_container_width=True)

    with t5:
        st.plotly_chart(chart_bits(audio, wm, hof), use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:40px 0 24px 0">
      <div style="font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;font-weight:800;color:#00d4ff;margin-bottom:10px">
        MARKIO READY
      </div>
      <div style="font-size:1.2rem;font-weight:800;color:#e5e9f7;margin-bottom:6px">
        Configure the sidebar and run <span style="color:#00d4ff">MARKIO Optimization</span>
      </div>
      <div style="color:#8890aa;font-size:.88rem">
        The engine will evolve a 3-gene watermark profile and build a true Pareto front.
      </div>
    </div>
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon">Signal</div>
        <div class="feat-title">Imperceptible Watermarking</div>
        <div class="feat-desc">MARKIO tunes spread-spectrum carriers to protect fidelity.
        SNR is maximised as a first-class objective, not hidden as a constraint.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">Recovery</div>
        <div class="feat-title">Robust Bit Recovery</div>
        <div class="feat-desc">Bit accuracy is evaluated after a Gaussian noise attack,
        pushing MARKIO toward profiles that survive real-world degradation.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">Tradeoff</div>
        <div class="feat-title">True Pareto Front</div>
        <div class="feat-desc">Non-dominated sorting + crowding distance — no weighted sum.
        The output is a full trade-off curve you can navigate.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">Source</div>
        <div class="feat-title">Real Audio Support</div>
        <div class="feat-desc">Upload your own WAV, AIFF, or FLAC file.
        Automatic mono conversion and resampling included.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
