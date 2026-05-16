import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, argparse, io

# Optional dependencies
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from scipy.signal import resample as scipy_resample
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# Terminal colors

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    CYAN    = "\033[36m";  MAGENTA = "\033[35m"
    YELLOW  = "\033[33m";  RED     = "\033[31m"
    GREEN   = "\033[32m";  WHITE   = "\033[37m"

    BCYAN   = "\033[96m";  BMAGENTA= "\033[95m"
    BYELLOW = "\033[93m";  BRED    = "\033[91m"
    BGREEN  = "\033[92m";  BWHITE  = "\033[97m"
    BBLUE   = "\033[94m"


def cprint(*parts, sep="", end="\n"):
    print(*parts, sep=sep, end=end, flush=True)

def _bar(frac, width=14, fill="█", empty="░", color=None):
    frac   = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    bar    = fill * filled + empty * (width - filled)
    if color:
        return f"{color}{bar}{C.RESET}"
    return bar

def _hbar(frac, width=30, color=None):
    b = _bar(frac, width=width, color=color)
    return f"{b} {C.BWHITE}{frac*100:5.1f}%{C.RESET}"

BANNER = (
    f"\n{C.BCYAN}{C.BOLD}"
    "  ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗██╗ ██████╗ \n"
    "  ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██║██╔═══██╗\n"
    "  ██╔████╔██║███████║██████╔╝█████╔╝ ██║██║   ██║\n"
    "  ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██║██║   ██║\n"
    "  ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗██║╚██████╔╝\n"
    "  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ \n"
    f"{C.RESET}"
    f"{C.BMAGENTA}"
    "  ╔══════════════════════════════════════════════════════════════════════════════╗\n"
    "  ║  MULTI-OBJECTIVE GENETIC ALGORITHM  ·  AUDIO WATERMARK OPTIMIZATION ENGINE  ║\n"
    "  ║  Chromosome: [alpha · band · n_carriers]  ·  True Pareto-Optimal Front      ║\n"
    "  ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    f"{C.RESET}"
)

def print_banner():
    cprint(BANNER)

def _section(title, icon="◈"):
    W = 74
    cprint(f"\n{C.BCYAN}{'─'*2} {C.BOLD}{C.BWHITE}{icon} {title} "
           f"{C.RESET}{C.BCYAN}{'─'*max(0, W-len(title)-4)}{C.RESET}")

def _kv(key, val, kw=24, key_c=C.CYAN, val_c=C.BWHITE):
    cprint(f"  {key_c}{key:<{kw}}{C.RESET}{val_c}{val}{C.RESET}")

def _ok(msg):   cprint(f"  {C.BGREEN}✔  {C.RESET}{msg}")
def _warn(msg): cprint(f"  {C.BYELLOW}⚠  {C.RESET}{msg}")
def _err(msg):  cprint(f"  {C.BRED}✘  {C.RESET}{msg}")
def _info(msg): cprint(f"  {C.BCYAN}╌  {C.RESET}{msg}")


# Audio I/O

def load_real_audio(source, target_sr):
    if not _HAS_SF:
        raise ImportError(
            "soundfile is required for real audio loading.\n"
            "  pip install soundfile"
        )

    if hasattr(source, "read"):
        # File-like input
        source.seek(0)
        sig, native_sr = sf.read(io.BytesIO(source.read()), dtype="float32", always_2d=False)
    else:
        sig, native_sr = sf.read(str(source), dtype="float32", always_2d=False)

    # Convert to mono
    if sig.ndim == 2:
        sig = sig.mean(axis=1)

    # Resample
    if native_sr != target_sr:
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for resampling.\n"
                "  pip install scipy"
            )
        n_out = int(len(sig) * target_sr / native_sr)
        sig   = scipy_resample(sig, n_out).astype(np.float32)

    # Normalize
    peak = np.max(np.abs(sig))
    if peak > 1e-6:
        sig /= peak

    return np.clip(sig, -1.0, 1.0), target_sr


# Signal processing

def generate_audio(duration=2.0, sample_rate=8000, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440  * t) +
           0.3 * np.sin(2 * np.pi * 880  * t) +
           0.1 * np.sin(2 * np.pi * 1320 * t) +
           0.08 * rng.standard_normal(len(t)))
    return sig / np.max(np.abs(sig)), sample_rate


def prepare_audio_source(source_mode, audio_file=None, sample_rate=8000, duration=2.0):
    if source_mode not in {"generate", "wav"}:
        raise ValueError("source_mode must be 'generate' or 'wav'.")

    if source_mode == "generate":
        if audio_file is not None:
            raise ValueError("Generated audio mode does not accept an audio file.")
        audio, sr = generate_audio(duration=duration, sample_rate=sample_rate)
        return audio, sr, "generated audio"

    if audio_file is None:
        raise ValueError("Real audio mode requires an audio file.")

    audio, sr = load_real_audio(audio_file, sample_rate)
    label = getattr(audio_file, "name", os.path.basename(str(audio_file)))
    return audio, sr, label


def generate_watermark(length=64, seed=7):
    return np.random.default_rng(seed).integers(0, 2, size=length).astype(float)


def _make_carrier(seg_len, band, carrier_seed):
    rng    = np.random.default_rng(carrier_seed)
    c      = rng.standard_normal(seg_len)
    window = max(1, int((1.0 - band) * seg_len * 0.15))
    if window > 1:
        c = np.convolve(c, np.ones(window) / window, mode="same")
    return c / (np.linalg.norm(c) + 1e-9)


def embed_watermark(audio, watermark, alpha, band=1.0, n_carriers=1):
    nc      = max(1, int(round(n_carriers)))
    n_bits  = len(watermark)
    seg_len = len(audio) // n_bits
    out     = audio.copy()
    for i, bit in enumerate(watermark):
        s, e   = i * seg_len, (i + 1) * seg_len
        symbol = 1.0 if bit else -1.0
        for k in range(nc):
            c = _make_carrier(e - s, band, i * 4 + k)
            out[s:e] += (alpha / nc) * symbol * c
    return np.clip(out, -1.0, 1.0)


def extract_watermark(original, received, n_bits, band=1.0, n_carriers=1):
    nc      = max(1, int(round(n_carriers)))
    seg_len = len(original) // n_bits
    bits    = np.zeros(n_bits)
    for i in range(n_bits):
        s, e = i * seg_len, (i + 1) * seg_len
        diff = received[s:e] - original[s:e]
        corr = sum(np.dot(diff, _make_carrier(e - s, band, i * 4 + k))
                   for k in range(nc))
        bits[i] = 1.0 if corr >= 0 else 0.0
    return bits


def compute_snr(original, watermarked):
    noise = watermarked - original
    pn    = np.mean(noise ** 2)
    return 10 * np.log10(np.mean(original ** 2) / pn) if pn > 1e-12 else 100.0


def compute_objectives(alpha, band, n_carriers, audio, watermark,
                        attack_std=0.05, attack_seed=42):
    wm       = embed_watermark(audio, watermark, alpha, band, n_carriers)
    snr      = compute_snr(audio, wm)
    attacked = wm + np.random.default_rng(attack_seed).normal(0, attack_std, len(wm))
    attacked = np.clip(attacked, -1.0, 1.0)
    ext      = extract_watermark(audio, attacked, len(watermark), band, n_carriers)
    acc      = float(np.mean(ext == watermark))
    return snr, acc


# Individual

class Individual:
    BOUNDS = np.array([[0.001, 0.15],
                       [0.0,   1.0],
                       [1.0,   3.0]])
    SIGMA0 = np.array([0.010, 0.05, 0.2])

    __slots__ = ("genes", "sigmas", "snr", "acc")

    def __init__(self, genes, sigmas=None):
        self.genes  = np.asarray(genes, dtype=float)
        self.sigmas = np.asarray(sigmas if sigmas is not None
                                  else self.SIGMA0.copy(), dtype=float)
        self.snr = self.acc = 0.0

    @property
    def alpha(self):      return self.genes[0]
    @property
    def band(self):       return self.genes[1]
    @property
    def n_carriers(self): return max(1, int(round(self.genes[2])))

    def evaluate(self, audio, watermark):
        self.snr, self.acc = compute_objectives(
            self.alpha, self.band, self.n_carriers, audio, watermark)

    def clone(self):
        ind = Individual(self.genes.copy(), self.sigmas.copy())
        ind.snr, ind.acc = self.snr, self.acc
        return ind

    def balanced_score(self, snr_ref=40.0):
        return min(self.snr / snr_ref, 1.0) + self.acc

    def __repr__(self):
        return (f"alpha={self.alpha:.4f}  band={self.band:.2f}  "
                f"nc={self.n_carriers}  SNR={self.snr:.1f}dB  "
                f"Acc={self.acc*100:.1f}%")


# Multi-objective selection

def _dominates(a_snr, a_acc, b_snr, b_acc):
    return (a_snr >= b_snr and a_acc >= b_acc) and (a_snr > b_snr or a_acc > b_acc)


def non_dominated_sort(pop):
    N         = len(pop)
    dom_count = np.zeros(N, int)
    dominated = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if _dominates(pop[i].snr, pop[i].acc, pop[j].snr, pop[j].acc):
                dominated[i].append(j)
            elif _dominates(pop[j].snr, pop[j].acc, pop[i].snr, pop[i].acc):
                dom_count[i] += 1
    fronts = [[i for i in range(N) if dom_count[i] == 0]]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
        k += 1
    ranks = np.zeros(N, int)
    for r, front in enumerate(fronts[:-1]):
        for i in front:
            ranks[i] = r
    return ranks, fronts[:-1]


def crowding_distance(pop, front_idx):
    n    = len(front_idx)
    dist = {i: 0.0 for i in front_idx}
    if n <= 2:
        for i in front_idx: dist[i] = float("inf")
        return dist
    for getter in [lambda p: p.snr, lambda p: p.acc]:
        vals = sorted(front_idx, key=lambda i: getter(pop[i]))
        dist[vals[0]]  = float("inf")
        dist[vals[-1]] = float("inf")
        span = getter(pop[vals[-1]]) - getter(pop[vals[0]])
        if span < 1e-9: continue
        for m in range(1, n - 1):
            dist[vals[m]] += (getter(pop[vals[m + 1]]) -
                              getter(pop[vals[m - 1]])) / span
    return dist


def all_crowding(pop, fronts):
    crowd = {}
    for front in fronts:
        crowd.update(crowding_distance(pop, front))
    return crowd


# Genetic algorithm

class GeneticAlgorithm:
    def __init__(self,
                 pop_size=40,
                 generations=80,
                 sbx_eta=5,
                 crossover_prob=0.85,
                 tournament_k_min=2,
                 tournament_k_max=6,
                 stagnation_patience=12,
                 injection_frac=0.30,
                 seed=0,
                 verbose=True):
        self.pop_size  = pop_size
        self.G         = generations
        self.eta       = sbx_eta
        self.cp        = crossover_prob
        self.k_min     = tournament_k_min
        self.k_max     = tournament_k_max
        self.patience  = stagnation_patience
        self.inj_frac  = injection_frac
        self.rng       = np.random.default_rng(seed)
        self.sigma_min = 1e-5
        self.tau       = 1.0 / np.sqrt(2 * 3)
        self.verbose   = verbose
        self._panel_lines = 0

    # Init
    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            genes = np.array([self.rng.uniform(*b) for b in Individual.BOUNDS])
            pop.append(Individual(genes))
        return pop

    # Operators
    def _tournament(self, pop, ranks, crowd, k):
        idx  = self.rng.choice(len(pop), size=k, replace=False)
        best = int(idx[0])
        for i in idx[1:]:
            if ranks[i] < ranks[best]:
                best = int(i)
            elif (ranks[i] == ranks[best] and
                  crowd.get(int(i), 0) > crowd.get(best, 0)):
                best = int(i)
        return pop[best]

    def _sbx(self, p1, p2):
        g1, g2 = p1.genes.copy(), p2.genes.copy()
        s1, s2 = p1.sigmas.copy(), p2.sigmas.copy()
        for d in range(3):
            if self.rng.random() > self.cp or abs(g1[d] - g2[d]) < 1e-9:
                continue
            u      = self.rng.random()
            b      = ((2 * u) ** (1 / (self.eta + 1)) if u <= 0.5
                      else (1 / (2 * (1 - u))) ** (1 / (self.eta + 1)))
            lo, hi = Individual.BOUNDS[d]
            g1[d]  = np.clip(0.5 * ((1 + b) * g1[d] + (1 - b) * g2[d]), lo, hi)
            g2[d]  = np.clip(0.5 * ((1 - b) * g1[d] + (1 + b) * g2[d]), lo, hi)
            s1[d]  = s2[d] = 0.5 * (p1.sigmas[d] + p2.sigmas[d])
        return Individual(g1, s1), Individual(g2, s2)

    def _mutate(self, ind):
        genes  = ind.genes.copy()
        sigmas = ind.sigmas.copy()
        for d in range(3):
            sigmas[d] = max(sigmas[d] * np.exp(self.tau * self.rng.standard_normal()),
                            self.sigma_min)
            lo, hi   = Individual.BOUNDS[d]
            genes[d] = np.clip(genes[d] + sigmas[d] * self.rng.standard_normal(),
                               lo, hi)
        return Individual(genes, sigmas)

    def _inject(self, pop, ranks, n_inject):
        order = sorted(range(len(pop)), key=lambda i: -ranks[i])
        for i in order[:n_inject]:
            genes  = np.array([self.rng.uniform(*b) for b in Individual.BOUNDS])
            pop[i] = Individual(genes)
        return pop

    def _k(self, gen):
        progress = gen / max(self.G - 1, 1)
        return max(2, round(self.k_min + progress * (self.k_max - self.k_min)))

    # Dashboard
    def _render_panel(self, gen, hof, front0_size, div, note, elapsed):
        pct  = (gen + 1) / self.G
        prog = _bar(pct, width=32, color=C.BCYAN)

        # Gene bars
        an = (hof.alpha - 0.001) / (0.15 - 0.001)
        bn = hof.band
        nn = (hof.genes[2] - 1.0) / 2.0

        inj_badge = (f"  {C.BMAGENTA}⚡ INJECT ×{note.split()[1]}{C.RESET}"
                     if note else "")

        lines = [
            f"  {C.DIM}{'─' * 72}{C.RESET}",
            (f"  {C.BOLD}{C.BCYAN}GEN {gen+1:>4} / {self.G:<4}{C.RESET}"
             f"  {prog}  {C.DIM}{elapsed:6.1f}s{C.RESET}{inj_badge}"),
            "",
            (f"  {C.BRED}SNR{C.RESET}  "
             f"{_bar(hof.snr / 60.0, width=18, color=C.BRED)}"
             f"  {C.BOLD}{C.BRED}{hof.snr:6.2f} dB{C.RESET}"
             f"       {C.BGREEN}ACC{C.RESET}  "
             f"{_bar(hof.acc, width=18, color=C.BGREEN)}"
             f"  {C.BOLD}{C.BGREEN}{hof.acc*100:5.1f}%{C.RESET}"),
            "",
            (f"  {C.CYAN}alpha{C.RESET}  {_bar(an, width=10, color=C.BCYAN)}"
             f"  {C.BCYAN}{hof.alpha:.5f}{C.RESET}"
             f"    {C.YELLOW}band{C.RESET}  {_bar(bn, width=10, color=C.BYELLOW)}"
             f"  {C.BYELLOW}{hof.band:.3f}{C.RESET}"
             f"    {C.MAGENTA}nc{C.RESET}  {_bar(nn, width=8, color=C.BMAGENTA)}"
             f"  {C.BMAGENTA}{hof.n_carriers}{C.RESET}"),
            "",
            (f"  {C.DIM}front₀ size={front0_size:<4}"
             f"  diversity={div:.5f}"
             f"  score={hof.balanced_score():.5f}{C.RESET}"),
            f"  {C.DIM}{'─' * 72}{C.RESET}",
        ]

        # Redraw panel
        if self._panel_lines:
            sys.stdout.write(f"\033[{self._panel_lines}A")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._panel_lines = len(lines)

    def _hist_snapshot(self, hist):
        return {k: v.copy() if hasattr(v, "copy") else v
                for k, v in hist.items()}

    # Evolution loop
    def iter_run(self, audio, watermark):
        pop = self._init_population()
        for ind in pop: ind.evaluate(audio, watermark)

        ranks, fronts = non_dominated_sort(pop)
        crowd         = all_crowding(pop, fronts)
        hof           = max(pop, key=lambda x: x.balanced_score()).clone()
        hof_score     = hof.balanced_score()
        no_imp        = 0

        hist = dict(hof_snr=[], hof_acc=[], hof_score=[],
                    front0_size=[], div=[], restarts=[])

        if self.verbose:
            _section("EVOLVING POPULATION", "⚛")
            cprint()

        t0 = time.perf_counter()

        for gen in range(self.G):
            ranks, fronts = non_dominated_sort(pop)
            crowd         = all_crowding(pop, fronts)

            for ind in pop:
                if ind.balanced_score() > hof.balanced_score():
                    hof = ind.clone()

            note = ""
            if hof.balanced_score() > hof_score + 1e-4:
                hof_score = hof.balanced_score()
                no_imp    = 0
            else:
                no_imp += 1

            if no_imp >= self.patience:
                n_inj = max(1, int(self.inj_frac * self.pop_size))
                pop   = self._inject(pop, ranks, n_inj)
                for ind in pop:
                    if ind.snr == 0.0: ind.evaluate(audio, watermark)
                no_imp = 0
                hist["restarts"].append(gen)
                note   = f"inject {n_inj}"

            front0_size = len(fronts[0]) if fronts else 0
            alphas      = np.array([p.alpha for p in pop])
            div         = float(np.std(alphas))

            hist["hof_snr"].append(hof.snr)
            hist["hof_acc"].append(hof.acc)
            hist["hof_score"].append(hof.balanced_score())
            hist["front0_size"].append(front0_size)
            hist["div"].append(div)

            if self.verbose:
                self._render_panel(gen, hof, front0_size, div, note,
                                   time.perf_counter() - t0)

            k       = self._k(gen)
            new_pop = [hof.clone()]
            while len(new_pop) < self.pop_size:
                p1     = self._tournament(pop, ranks, crowd, k)
                p2     = self._tournament(pop, ranks, crowd, k)
                c1, c2 = self._sbx(p1, p2)
                c1     = self._mutate(c1); c1.evaluate(audio, watermark)
                c2     = self._mutate(c2); c2.evaluate(audio, watermark)
                new_pop.extend([c1, c2])
            pop = new_pop[:self.pop_size]

            yield gen + 1, hof.clone(), self._hist_snapshot(hist), note, None, None

        ranks, fronts = non_dominated_sort(pop)
        pareto_front  = [pop[i] for i in fronts[0]] if fronts else []

        if self.verbose:
            cprint()
            _section("HALL OF FAME", "★")
            _kv("alpha  (embed strength)", f"{hof.alpha:.6f}",   key_c=C.BCYAN)
            _kv("band   (carrier BW)",     f"{hof.band:.4f}",    key_c=C.BYELLOW)
            _kv("n_carriers  (per bit)",   str(hof.n_carriers),  key_c=C.BMAGENTA)
            _kv("SNR",                     f"{hof.snr:.3f} dB",  key_c=C.BRED)
            _kv("Bit Accuracy",            f"{hof.acc*100:.2f}%",key_c=C.BGREEN)
            _kv("Balanced Score",          f"{hof.balanced_score():.5f}")
            _kv("Pareto front size",       str(len(pareto_front)))
            _kv("Diversity injections",    str(len(hist["restarts"])))
            _kv("Wall time",               f"{time.perf_counter()-t0:.2f}s")

        yield self.G, hof.clone(), self._hist_snapshot(hist), "__done__", pop, pareto_front

    def run(self, audio, watermark):
        for gen, hof, hist, note, final_pop, pareto_front in self.iter_run(audio, watermark):
            if note == "__done__":
                return hof, pareto_front, final_pop, hist
        raise RuntimeError("Genetic algorithm did not produce a final result.")


# Plots

def plot_results(audio, watermark, hof, pareto_front, final_pop, hist, sr,
                 out_dir="./outputs"):
    wm_opt    = embed_watermark(audio, watermark, hof.alpha, hof.band, hof.n_carriers)
    final_snr = compute_snr(audio, wm_opt)

    BG, PANEL    = "#12141f", "#1c1f30"
    CYAN, CORAL  = "#00d4ff", "#ff6b6b"
    GREEN, AMBER = "#6dffa0", "#ffcc44"
    PURP, TEXT   = "#b57bee", "#dde1f0"
    SUBTEXT      = "#8890aa"

    os.makedirs(out_dir, exist_ok=True)

    def style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=6, fontweight="bold")
        ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=9)
        ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=9)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2d40")
        ax.grid(color="#22263a", lw=0.5, ls="--")

    def new_fig(wide=False):
        fig, ax = plt.subplots(figsize=(9 if wide else 5, 4))
        fig.patch.set_facecolor(BG)
        return fig, ax

    def save(fig, fname):
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        _ok(f"Saved  {C.DIM}{path}{C.RESET}")

    t   = np.linspace(0, len(audio) / sr, len(audio))
    gen = np.arange(1, len(hist["hof_snr"]) + 1)

    fig, ax = new_fig()
    ax.plot(t[:2000], audio[:2000], color=CYAN, lw=0.7)
    style(ax, "Original Audio", "Time (s)", "Amplitude")
    save(fig, "01_original_audio.png")

    fig, ax = new_fig()
    ax.plot(t[:2000], wm_opt[:2000], color=CORAL, lw=0.7)
    style(ax, f"Watermarked  alpha={hof.alpha:.4f}", "Time (s)", "Amplitude")
    save(fig, "02_watermarked.png")

    fig, ax = new_fig()
    ax.plot(t[:2000], (wm_opt - audio)[:2000], color=GREEN, lw=0.7)
    style(ax, f"Embedded Noise  SNR={final_snr:.1f} dB", "Time (s)", "Amplitude")
    save(fig, "03_embedded_noise.png")

    fig, ax = new_fig()
    ext = extract_watermark(audio, wm_opt, len(watermark), hof.band, hof.n_carriers)
    x   = np.arange(len(watermark))
    ax.bar(x, watermark, width=0.8, color=CYAN,  alpha=0.55, label="Original")
    ax.bar(x, ext,       width=0.4, color=CORAL, alpha=0.9,  label="Extracted")
    clean_acc = np.mean(ext == watermark) * 100
    style(ax, f"Bit Recovery (clean) {clean_acc:.0f}%", "Bit index", "Value")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    save(fig, "04_bit_recovery.png")

    fig, ax = new_fig(wide=True)
    ax.plot(gen, hist["hof_snr"], color=CORAL, lw=1.6, label="HoF SNR (dB)")
    axb = ax.twinx()
    axb.plot(gen, [a * 100 for a in hist["hof_acc"]], color=GREEN, lw=1.6, ls="--", label="HoF Acc (%)")
    for rg in hist["restarts"]: ax.axvline(rg + 1, color=PURP, lw=0.9, ls=":", alpha=0.7)
    style(ax, "HoF Objectives over Generations", "Generation", "SNR (dB)")
    ax.tick_params(axis="y", colors=CORAL, labelsize=9)
    axb.tick_params(axis="y", colors=GREEN, labelsize=9)
    axb.set_ylabel("Accuracy (%)", color=GREEN, fontsize=9)
    axb.set_facecolor(PANEL)
    for sp in axb.spines.values(): sp.set_edgecolor("#2a2d40")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="lower right")
    axb.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="upper right")
    save(fig, "05_hof_objectives.png")

    fig, ax = new_fig()
    ax.plot(gen, hist["front0_size"], color=AMBER, lw=1.5, label="Pareto front size")
    ax.plot(gen, [d * 100 for d in hist["div"]], color=PURP, lw=1.5, ls="--", label="Pop diversity x100")
    for rg in hist["restarts"]: ax.axvline(rg + 1, color=PURP, lw=0.9, ls=":", alpha=0.5)
    style(ax, "Pareto Front Size & Diversity", "Generation", "")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    save(fig, "06_pareto_diversity.png")

    fig, ax = new_fig()
    genes_norm  = [(hof.alpha-0.001)/(0.15-0.001), hof.band, (hof.genes[2]-1.0)/2.0]
    gene_labels = [f"alpha={hof.alpha:.4f}", f"band={hof.band:.2f}", f"nc={hof.n_carriers}"]
    bars = ax.barh(gene_labels, genes_norm, color=[CYAN, AMBER, GREEN], alpha=0.85)
    style(ax, "HoF Chromosome (normalised)", "Value [0,1]", "")
    for bar, val in zip(bars, genes_norm):
        ax.text(min(val+0.02, 0.95), bar.get_y()+bar.get_height()/2,
                f"{val:.2f}", color=TEXT, fontsize=9, va="center")
    ax.set_xlim(0, 1.1)
    save(fig, "07_hof_chromosome.png")

    fig, ax = new_fig(wide=True)
    ax.scatter([p.snr for p in final_pop], [p.acc*100 for p in final_pop],
               color=SUBTEXT, s=20, alpha=0.4, label="Final population", zorder=2)
    pf_sorted = sorted(pareto_front, key=lambda x: x.snr)
    ax.scatter([p.snr for p in pareto_front], [p.acc*100 for p in pareto_front],
               color=AMBER, s=50, zorder=3, label="Pareto front (rank 0)")
    ax.plot([p.snr for p in pf_sorted], [p.acc*100 for p in pf_sorted], color=AMBER, lw=1, alpha=0.5)
    ax.scatter([hof.snr], [hof.acc*100], color=CYAN, s=120, marker="*", zorder=4, label="HoF (best balanced)")
    style(ax, "Final Pareto Front - SNR vs Robustness", "SNR (dB)", "Bit Accuracy (%)")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)
    save(fig, "08_final_pareto_front.png")

    fig, ax = new_fig()
    sweep_a = np.linspace(0.001, 0.15, 50)
    sw_snr, sw_acc = [], []
    for a in sweep_a:
        s, c = compute_objectives(a, hof.band, hof.n_carriers, audio, watermark)
        sw_snr.append(s); sw_acc.append(c*100)
    ax.plot(sweep_a, sw_snr, color=CORAL, lw=1.4, label="SNR (dB)")
    axr = ax.twinx()
    axr.plot(sweep_a, sw_acc, color=GREEN, lw=1.4, ls="--", label="Acc (%)")
    ax.axvline(hof.alpha, color=CYAN, lw=1.2, ls="--", label="HoF alpha")
    style(ax, f"alpha Sweep (band={hof.band:.2f}, nc={hof.n_carriers})", "alpha", "SNR (dB)")
    ax.tick_params(axis="y", colors=CORAL, labelsize=9)
    axr.tick_params(axis="y", colors=GREEN, labelsize=9)
    axr.set_ylabel("Accuracy (%)", color=GREEN, fontsize=9)
    axr.set_facecolor(PANEL)
    for sp in axr.spines.values(): sp.set_edgecolor("#2a2d40")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    save(fig, "09_alpha_sweep.png")

    fig, ax = new_fig()
    sweep_b = np.linspace(0.0, 1.0, 30)
    sb_snr, sb_acc = [], []
    for b in sweep_b:
        s, c = compute_objectives(hof.alpha, b, hof.n_carriers, audio, watermark)
        sb_snr.append(s); sb_acc.append(c*100)
    ax.plot(sweep_b, sb_snr, color=CORAL, lw=1.4, label="SNR (dB)")
    axr = ax.twinx()
    axr.plot(sweep_b, sb_acc, color=GREEN, lw=1.4, ls="--", label="Acc (%)")
    ax.axvline(hof.band, color=CYAN, lw=1.2, ls="--", label="HoF band")
    style(ax, f"Band Sweep (alpha={hof.alpha:.4f}, nc={hof.n_carriers})", "band", "SNR (dB)")
    ax.tick_params(axis="y", colors=CORAL, labelsize=9)
    axr.tick_params(axis="y", colors=GREEN, labelsize=9)
    axr.set_ylabel("Accuracy (%)", color=GREEN, fontsize=9)
    axr.set_facecolor(PANEL)
    for sp in axr.spines.values(): sp.set_edgecolor("#2a2d40")
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    save(fig, "10_band_sweep.png")


# CLI

PRESETS = {
    "quick":    dict(pop_size=20, generations=30,  sbx_eta=5,  k_min=2, k_max=4,  patience=8,  inj_frac=0.30),
    "balanced": dict(pop_size=40, generations=80,  sbx_eta=5,  k_min=2, k_max=6,  patience=12, inj_frac=0.30),
    "thorough": dict(pop_size=60, generations=120, sbx_eta=10, k_min=2, k_max=8,  patience=18, inj_frac=0.25),
}


def build_parser():
    p = argparse.ArgumentParser(
        prog="audio_watermarking_ga",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{C.BCYAN}{C.BOLD}  MO-GA Audio Watermark Optimizer{C.RESET}\n"
            f"{C.DIM}  Evolves a 3-gene chromosome [alpha x band x n_carriers]\n"
            f"  to jointly maximise SNR and Bit Accuracy (true Pareto front).{C.RESET}\n\n"
            f"{C.BYELLOW}  Examples:{C.RESET}\n"
            f"    {C.BWHITE}python audio_watermarking_ga.py --source generate{C.RESET}\n"
            f"    {C.BWHITE}python audio_watermarking_ga.py --source wav --audio track.wav --sr 16000{C.RESET}\n"
            f"    {C.BWHITE}python audio_watermarking_ga.py --source generate --preset thorough --bits 128{C.RESET}\n"
            f"    {C.BWHITE}python audio_watermarking_ga.py --source generate --pop 80 --gens 200 --seed 1337{C.RESET}\n"
            f"    {C.BWHITE}python audio_watermarking_ga.py --source generate --no-plots --quiet{C.RESET}\n"
        ),
    )

    aud = p.add_argument_group(f"{C.CYAN}audio{C.RESET}")
    aud.add_argument("--source", choices=["generate", "wav"], required=True,
                     help="Required audio source: generate synthetic audio or load a real audio file.")
    aud.add_argument("--audio", metavar="FILE",
                     help="Input audio file (WAV / AIFF / FLAC). Required with --source wav.")
    aud.add_argument("--sr", type=int, default=8000, metavar="HZ",
                     choices=[8000, 16000, 22050, 44100],
                     help="Target sample rate Hz  (default: 8000)")
    aud.add_argument("--duration", type=float, default=2.0, metavar="S",
                     help="Synthetic signal duration in seconds  (default: 2.0)")
    aud.add_argument("--bits", type=int, default=64, metavar="N",
                     choices=[16, 32, 64, 128],
                     help="Watermark length in bits  (default: 64)")

    ga = p.add_argument_group(f"{C.CYAN}genetic algorithm{C.RESET}")
    ga.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                    help="Named preset: quick / balanced / thorough")
    ga.add_argument("--pop",  type=int,   default=None, metavar="N",
                    help="Population size  (overrides preset)")
    ga.add_argument("--gens", type=int,   default=None, metavar="N",
                    help="Generations  (overrides preset)")
    ga.add_argument("--eta",  type=int,   default=None, metavar="N",
                    help="SBX eta crossover parameter")
    ga.add_argument("--kmin", type=int,   default=None, metavar="K",
                    help="Tournament size minimum")
    ga.add_argument("--kmax", type=int,   default=None, metavar="K",
                    help="Tournament size maximum")
    ga.add_argument("--patience", type=int,   default=None, metavar="N",
                    help="Stagnation patience before injection")
    ga.add_argument("--inj-frac", type=float, default=None, metavar="F",
                    help="Fraction of population to inject on stagnation  (0.0-1.0)")
    ga.add_argument("--seed", type=int, default=0, metavar="N",
                    help="RNG seed  (default: 0)")

    out = p.add_argument_group(f"{C.CYAN}output{C.RESET}")
    out.add_argument("--output-dir", default="./outputs", metavar="DIR",
                     help="Directory for saved plots  (default: ./outputs)")
    out.add_argument("--no-plots", action="store_true",
                     help="Skip generating matplotlib plots")
    out.add_argument("--quiet", action="store_true",
                     help="Suppress banner and live progress display")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.source == "wav" and not args.audio:
        parser.error("--source wav requires --audio FILE.")
    if args.source == "generate" and args.audio:
        parser.error("--source generate cannot be used with --audio.")

    if not args.quiet:
        print_banner()

    # GA config
    base = PRESETS[args.preset] if args.preset else PRESETS["balanced"]
    cfg  = dict(
        pop_size            = args.pop      or base["pop_size"],
        generations         = args.gens     or base["generations"],
        sbx_eta             = args.eta      or base["sbx_eta"],
        crossover_prob      = 0.85,
        tournament_k_min    = args.kmin     or base["k_min"],
        tournament_k_max    = args.kmax     or base["k_max"],
        stagnation_patience = args.patience or base["patience"],
        injection_frac      = args.inj_frac or base["inj_frac"],
        seed                = args.seed,
        verbose             = not args.quiet,
    )

    # Load audio
    if not args.quiet:
        _section("SYSTEM INIT", "◈")

    if args.source == "wav":
        if not os.path.isfile(args.audio):
            _err(f"File not found: {args.audio}")
            sys.exit(1)
        if not args.quiet:
            _info(f"Loading  {C.BWHITE}{args.audio}{C.RESET}  ->  target SR {args.sr} Hz")
        audio, sr, src_label = prepare_audio_source("wav", args.audio, args.sr, args.duration)
    else:
        if not args.quiet:
            _info(f"Generating synthetic signal  "
                  f"{C.BWHITE}{args.duration}s{C.RESET}  @  {args.sr} Hz  seed=42")
        audio, sr, src_label = prepare_audio_source("generate", None, args.sr, args.duration)

    watermark = generate_watermark(length=args.bits)

    if not args.quiet:
        _kv("Audio source", src_label)
        _kv("Samples",      f"{len(audio):,}  ({len(audio)/sr:.3f}s)")
        _kv("Sample rate",  f"{sr} Hz")
        _kv("Watermark",    f"{args.bits} bits  (seed=7)")
        cprint()
        _section("BASELINE", "▸")

    b_snr, b_acc = compute_objectives(0.05, 1.0, 1, audio, watermark)

    if not args.quiet:
        _kv("alpha=0.05  band=1.0  nc=1",
            f"SNR {C.BRED}{b_snr:.2f} dB{C.RESET}  "
            f"Acc {C.BGREEN}{b_acc*100:.1f}%{C.RESET}")
        cprint()
        _section("GA CONFIG", "▸")
        _kv("Population",  str(cfg["pop_size"]))
        _kv("Generations", str(cfg["generations"]))
        _kv("SBX eta",     str(cfg["sbx_eta"]))
        _kv("Tournament",  f"k in [{cfg['tournament_k_min']}, {cfg['tournament_k_max']}]")
        _kv("Patience",    str(cfg["stagnation_patience"]))
        _kv("Inj. frac",   f"{cfg['injection_frac']:.0%}")
        _kv("Seed",        str(cfg["seed"]))
        _kv("Preset",      args.preset if args.preset else "balanced (default)")
        cprint()

    # Run GA
    ga  = GeneticAlgorithm(**cfg)
    hof, pareto_front, final_pop, hist = ga.run(audio, watermark)

    # Compare baseline
    if not args.quiet:
        cprint()
        _section("DELTA VS BASELINE", "D")
        d_snr = hof.snr - b_snr
        d_acc = (hof.acc - b_acc) * 100
        _kv("DSNR", f"{C.BGREEN if d_snr>=0 else C.BRED}{d_snr:+.3f} dB{C.RESET}")
        _kv("DAcc", f"{C.BGREEN if d_acc>=0 else C.BRED}{d_acc:+.2f}%{C.RESET}")
        _kv("Injections", str(len(hist["restarts"])))

    # Save plots
    if not args.no_plots:
        if not args.quiet:
            cprint()
            _section("SAVING PLOTS", "o")
        plot_results(audio, watermark, hof, pareto_front, final_pop,
                     hist, sr, out_dir=args.output_dir)

    if not args.quiet:
        cprint()
        cprint(f"  {C.BGREEN}{C.BOLD}DONE{C.RESET}  "
               f"{C.DIM}Results saved to {args.output_dir}/{C.RESET}")
        cprint()


if __name__ == "__main__":
    main()
