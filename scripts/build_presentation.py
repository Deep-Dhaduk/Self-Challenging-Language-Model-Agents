"""Build the editable PowerPoint, a PDF visual edition, and presenter notes.

Requires python-pptx and Pillow. The local build can use the isolated dependency
directory outputs/pptx_dependencies; notebook/training dependencies are unchanged.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "outputs" / "pptx_dependencies"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = ROOT / "slides"
PREVIEWS = ROOT / "outputs" / "presentation_previews"
OUT.mkdir(exist_ok=True)
PREVIEWS.mkdir(parents=True, exist_ok=True)
W, H, DPI = 13.333333, 7.5, 144
NAVY, INK, TEAL = "10283F", "142D43", "087F82"
MUTED, LIGHT, WHITE, GOLD = "526679", "F0F5F8", "FFFFFF", "E6AE49"
BLUE, GREEN = "597DA8", "2D8C7C"
PAPER = "https://arxiv.org/abs/2506.01716"
SURVEY = "https://arxiv.org/abs/2607.13104"
REPO = "https://github.com/Deep-Dhaduk/Self-Challenging-Language-Model-Agents"
MEDIUM = "https://medium.com/@deepbharatbhai.dhaduk/can-ai-agents-learn-by-creating-their-own-challenges-dd7cdbf376e7"
COLAB = "https://colab.research.google.com/github/Deep-Dhaduk/Self-Challenging-Language-Model-Agents/blob/main/notebooks/self_challenging_agents.ipynb"
SOURCE_URLS = {
    "paper": PAPER,
    "survey": SURVEY,
    "self_instruct": "https://arxiv.org/abs/2212.10560",
    "react": "https://arxiv.org/abs/2210.03629",
    "reflexion": "https://arxiv.org/abs/2303.11366",
    "codeact": "https://arxiv.org/abs/2402.01030",
    "tau_bench": "https://arxiv.org/abs/2406.12045",
    "cpu_results": REPO + "/tree/main/results/cpu_demo",
}
DATA = json.loads((ROOT / "results/cpu_demo/summary.json").read_text())
ABLATIONS = json.loads((ROOT / "results/cpu_demo/filter_ablation.json").read_text())


def font(size, bold=False):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [windows / ("arialbd.ttf" if bold else "arial.ttf"),
                  Path("/usr/share/fonts/truetype/msttcorefonts") / ("Arial_Bold.ttf" if bold else "Arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), round(size * DPI / 72))
    raise RuntimeError("Install Arial or DejaVu Sans to build slide previews")


class Slide:
    def __init__(self, title, section, seconds, notes, sources=(), dark=False):
        self.title, self.section, self.seconds = title, section, seconds
        self.notes, self.sources, self.dark = notes.strip(), list(sources), dark
        self.elements = []
        self.box(0, 0, W, H, NAVY if dark else WHITE, radius=False)
        if not dark:
            self.box(.0, .0, W, .075, TEAL, radius=False)
            self.text(section.upper(), .55, .28, 12, .25, 10, TEAL, True)
            self.text(title, .55, .83, 12.2, .8, 30, INK, True)

    def box(self, x, y, w, h, fill, line=None, radius=True):
        self.elements.append(dict(kind="box", x=x, y=y, w=w, h=h, fill=fill, line=line, radius=radius))

    def line(self, x1, y1, x2, y2, color=TEAL, width=2):
        self.elements.append(dict(kind="line", x=x1, y=y1, w=x2-x1, h=y2-y1, color=color, width=width))

    def text(self, value, x, y, w, h, size=20, color=INK, bold=False, align="left", link=None):
        face = font(size, bold)
        lines = []
        for paragraph in str(value).split("\n"):
            words = paragraph.split()
            current = ""
            for word in words:
                proposed = (current + " " + word).strip()
                if current and face.getlength(proposed) > w * DPI:
                    lines.append(current)
                    current = word
                else:
                    current = proposed
            lines.append(current)
        needed = len(lines) * size * 1.2 / 72
        if needed > h + .04:
            raise ValueError(f"Text overflow on {self.title}: {value!r} needs {needed:.2f}in, has {h:.2f}in")
        self.elements.append(dict(kind="text", text="\n".join(lines), x=x, y=y, w=w, h=h,
                                  size=size, color=color, bold=bold, align=align, link=link))

    def card(self, x, y, w, h, label, body, accent=TEAL, size=20):
        self.box(x, y, w, h, LIGHT)
        self.box(x, y, .06, h, accent, radius=False)
        self.text(label, x+.22, y+.18, w-.44, .55, 19, accent, True)
        self.text(body, x+.22, y+.87, w-.44, h-1.0, size)

    def footer(self, number, total):
        color = "B8CBD9" if self.dark else MUTED
        self.text("INDIVIDUAL SHORT STORY ASSIGNMENT", .55, 7.09, 8, .2, 9, color)
        self.text(f"{number:02d} / {total:02d}", 11.7, 7.04, 1.05, .26, 11, color, align="right")

    def source(self, label, url):
        self.text(label, .55, 6.65, 12.1, .25, 10, MUTED, link=url)


slides = []


def add(title, section, seconds, notes, sources=(), dark=False):
    result = Slide(title, section, seconds, notes, sources, dark)
    slides.append(result)
    return result


s = add("Can AI agents learn by creating their own challenges?", "Opening", 35, """
This presentation examines how an agent can obtain useful practice experience and how we can tell whether that experience helped. The selected research case study is connected to a broader taxonomy, then to a small companion implementation. Keep published research, measured CPU results, and two unsuccessful LLM attempts separate. Neither LLM attempt reached adapter training. A native-function-call configuration is prepared but has no pretrained-model results yet. Follow the story from a customer request to task validation, policy updates, and separate evaluation. The planned timing is about sixteen minutes; rehearse it before recording because reading speed and pauses determine the actual duration.
""", (PAPER,), dark=True)
s.text("RESEARCH CASE STUDY + EXECUTABLE COMPANION PROJECT", .7, .58, 12, .4, 12, "8DDAD5", True)
s.text("Can AI agents learn by\ncreating their own challenges?", .7, 1.55, 12, 2.0, 42, WHITE, True)
s.text("Self-Challenging Language Model Agents\nZhou et al. | arXiv:2506.01716 | June 2025", .75, 4.02, 11.7, .9, 23, "DCE8EF")
s.text("Deep Dhaduk  |  Individual assignment", .75, 5.62, 10, .4, 18, WHITE)
s.text("Paper findings • CPU results • Two LLM attempts + next check", .75, 6.23, 11.8, .35, 14, "8DDAD5")

s = add("A return request is a workflow", "01 / Motivation", 55, """
Start with a familiar example. A customer wants to return headphones and receive the refund through the original payment method. The assistant has to identify the customer, find the intended order, inspect it, and apply the correct change. A fluent final sentence does not establish that these actions happened. The important evidence is the resulting state and the information returned by tools. Now think about preparing a training task for this workflow. We need a clear request, a feasible starting state, and a way to evaluate the outcome. If the supposed order does not exist, the agent cannot solve the task. If the checker ignores the refund method, the agent can receive misleading feedback. This motivates examining the entire practice pipeline, not just the final response.
""")
s.box(.6, 1.85, 12.1, 1.05, LIGHT)
s.text('“Return my headphones. Refund the original payment method.”', .9, 2.15, 11.5, .45, 23, INK, True)
for i, (a, b) in enumerate([("Identify", "Find the customer"), ("Inspect", "Locate and read order"), ("Act", "Apply the right change"), ("Check", "Verify resulting state")]):
    s.card(.6+i*3.08, 3.5, 2.88, 1.9, a, b, size=18)
s.text("Training examples also need feasible tasks and dependable feedback.", .75, 5.9, 11.9, .5, 23, TEAL, True)

s = add("What changes when an agent improves?", "02 / Taxonomy", 60, """
The recent survey provides an organizing distinction: the underlying model can change, or the surrounding scaffold can change. A scaffold is the collection of instructions, memory, tools, and control logic around the model. This avoids treating every improvement mechanism as the same kind of training. For this presentation, use a practical diagnostic: ask what persists after the experience. Is there a weight update, a revised instruction, a stored lesson, or an altered workflow? Then ask which test would detect its contribution. A useful comparison holds other factors steady so that additional attempts or a different model are not mistaken for the effect of a particular mechanism. The taxonomy is a map for reasoning about designs; it is not a ranking of the rows.
""", (SURVEY,))
s.card(.6, 1.85, 5.87, 3.75, "MODEL PARAMETERS", "Fine-tuning changes the model.\n\nMeasure gains on separate tasks.", size=23)
s.card(6.72, 1.85, 5.98, 3.75, "OPERATIONAL SCAFFOLD", "Prompts • Memory • Tools\nControl logic\n\nMeasure the changed component.", size=23)
s.text("Ask: what is updated, what feedback drives it, and what evidence survives evaluation?", .72, 5.95, 11.9, .5, 19, TEAL, True)
s.source("Taxonomy context: Ren et al. (2026), Self-Improvements in Modern Agentic Systems", SURVEY)

s = add("Four techniques clarify the research landscape", "03 / Related approaches", 65, """
These approaches illuminate different design choices. Self-Instruct generates instruction examples and filters them before fine-tuning. ReAct interleaves reasoning and actions so that external observations can inform execution. Reflexion retains textual feedback in memory for later attempts without changing model weights. CodeAct represents actions using executable code. Read each method in terms of its purpose, rather than treating their scores as entries on one shared leaderboard. Their original experiments differ in tasks, models, and evaluation settings. The engineering comparison on this slide therefore describes mechanisms. For a fair performance ranking, we would need matched tasks, a common evaluation protocol, and comparable resource budgets. This distinction is especially important when an article draws from several papers to explain one research area.
""", (SOURCE_URLS["self_instruct"], SOURCE_URLS["react"], SOURCE_URLS["reflexion"], SOURCE_URLS["codeact"]))
rows = [("Self-Instruct", "Generate and filter instruction examples", "Training data"),
        ("ReAct", "Interleave reasoning, actions, observations", "Execution"),
        ("Reflexion", "Retain textual lessons across attempts", "Memory"),
        ("CodeAct", "Represent actions as executable code", "Tool interface")]
for i, (name, description, target) in enumerate(rows):
    y = 1.86+i*1.02
    s.box(.6, y, 12.1, .86, LIGHT if i%2 == 0 else "F8FAFC")
    s.text(name, .84, y+.23, 2.1, .38, 20, TEAL, True)
    s.text(description, 3.02, y+.23, 6.5, .38, 19)
    s.text(target, 10.05, y+.25, 2.3, .35, 17, MUTED)
s.text("Compare mechanisms here; the original benchmark scores are not directly interchangeable.", .76, 6.19, 11.8, .35, 16, MUTED)
s.source("Sources: Wang et al. (Self-Instruct); Yao et al. (ReAct); Shinn et al. (Reflexion); Wang et al. (CodeAct). Links in notes.", SOURCE_URLS["self_instruct"])

s = add("The paper separates two roles", "04 / Selected paper", 50, """
Think of this separation as a way to make the practice process inspectable. One responsibility concerns preparing experience, and the other concerns acting on a request. An engineering review can inspect the material before it affects the executor. A useful question for the audience is where a defect would enter: during creation of the request, evaluation of the outcome, or selection of actions. Separating those responsibilities makes that question easier to investigate. The educational companion will expose corresponding artifacts so that an example can be followed across the pipeline. This is an architectural interpretation, not a claim that the system can improve indefinitely or operate without a designed environment.
""", (PAPER,))
s.card(.7, 2.0, 5.5, 2.85, "CHALLENGER", "Explore tools.\nAuthor a practice task.", size=26)
s.card(7.1, 2.0, 5.5, 2.85, "EXECUTOR", "Attempt accepted tasks.\nLearn from feedback.", size=26)
s.text("→", 6.27, 2.93, .7, .7, 36, TEAL, True, align="center")
s.text("A practice pipeline needs quality control before training.", .75, 5.6, 11.8, .65, 26, TEAL, True)
s.source("Paper: Zhou et al. (2025), §4. Original diagram for this presentation.", PAPER)

s = add("Code-as-Task makes practice testable", "05 / Task validation", 55, """
For our engineering discussion, imagine reviewing a task package before accepting it. Can we demonstrate that the expected behavior is possible? Can we show that plausible wrong outcomes are rejected? These are useful review questions because a checker can run successfully while expressing the wrong condition. The companion implementation turns these questions into an inspectable fixture. Its check specification is interpreted by trusted code, rather than executing arbitrary generated Python. That implementation choice is a deliberate simplification. It lets the demonstration focus on the quality of feedback. Even then, a small collection of examples is incomplete evidence: an untested wrong outcome may still pass, and natural-language intent may remain ambiguous.
""", (PAPER,))
for i, (label, body) in enumerate([("Instruction", "Requested outcome"), ("Verifier", "Executable success check"), ("Example solution", "One feasible solution"), ("Three failure cases", "Wrong outcomes to reject")]):
    x = .65+(i%2)*6.25
    y = 1.9+(i//2)*1.64
    s.card(x, y, 6.0, 1.4, label, body, size=18)
s.box(.7, 5.45, 11.9, .72, "DEF2EF")
s.text("Keep tasks when the example passes and failure cases fail.", .95, 5.65, 11.35, .4, 22, TEAL, True)
s.source("Paper: Zhou et al. (2025), §4.2. Finite tests provide evidence, not a correctness proof.", PAPER)

s = add("A concrete task in our retail sandbox", "06 / Worked example", 65, """
Walk through this example slowly. The request identifies the product and refund choice. The executor begins with the customer email, asks for orders, inspects the relevant one, and applies a return. The reference sequence belongs to the task package and is not placed in the executor's prompt. Next, consider three ways this can go wrong. Doing nothing should fail. Stopping after inspection should fail. Returning to the wrong payment destination should fail. There is one more condition worth emphasizing: other orders must remain unchanged. That condition makes the evaluator more demanding than checking for a return call. A complete demonstration should show both the conversation and the final state. All customers, orders, and actions here are simulated and reset between attempts.
""")
s.box(.65, 1.82, 12.0, .87, LIGHT)
s.text("Return the headphones; refund to original payment; preserve other orders.", .9, 2.06, 11.5, .4, 21, INK, True)
s.card(.65, 3.02, 5.8, 2.85, "REFERENCE WORKFLOW", "1  Look up customer\n2  List and inspect orders\n3  Return the intended order", size=22)
s.card(6.7, 3.02, 5.95, 2.85, "FAILURE PROBES", "No action\nIncomplete workflow\nWrong refund method", accent="A36C15", size=22)
s.source("Companion task/environment implementation. Reference traces are hidden from the executor.", REPO + "/blob/main/sca/tasks.py")

s = add("Paper results: average Pass@1", "07 / Published evidence", 65, """
Read this as a comparison within the authors' experiment. The three values represent different settings, so the stronger-teacher setting should not be described as autonomous improvement with no stronger model. Percentage-point changes and relative percentage changes also mean different things. For example, the difference between two percentages is an absolute percentage-point gain; dividing that change by the starting value yields a relative gain. More generally, an average can hide variation among domains, so a reviewer should consult the domain breakdown before drawing deployment conclusions. This slide is a historical paper result. It is not a current leaderboard and it is not the output of our repository. The companion results appear later with their own experiment identity.
""", (PAPER,))
for i, (label, value, color) in enumerate([("Untuned baseline", 12.0, BLUE), ("Self-improvement", 23.5, TEAL), ("Teacher distillation", 32.2, "2D5E86")]):
    y = 2.05+i*1.09
    s.text(label, .77, y+.18, 3.5, .5, 22, INK, i==1)
    s.box(4.35, y+.1, 6.8, .58, LIGHT)
    s.box(4.35, y+.1, 6.8*value/40, .58, color)
    s.text(f"{value:.1f}%", 11.42, y+.13, 1.13, .5, 23, color, True)
s.text("Llama-3.1-8B-Instruct • Four environments • M3ToolEval + TauBench", .8, 5.57, 11.8, .45, 20, MUTED)
s.source("Paper Table 1. Self-improvement uses successful self-rollouts; distillation uses a stronger teacher.", PAPER)

s = add("Paper ablations reveal engineering trade-offs", "08 / Published analysis", 55, """
An ablation should tell us which design decision changes an outcome. The first result on this slide concerns acceptance of generated practice material; the second concerns a task-success measurement in a particular experiment. They use different denominators and must not be plotted as though they measure the same thing. From an engineering perspective, stricter filtering introduces a quality-versus-yield question. If accepted material becomes scarce, the cost per useful example can rise. A more involved optimizer introduces another question: does its gain justify infrastructure and tuning effort? In a small coursework implementation, a simple inspectable baseline is valuable before expanding complexity. These are practical interpretations of the evidence, not new experimental findings.
""", (PAPER,))
s.card(.7, 1.95, 5.8, 3.7, "TASK QUALITY", "5.2% retained\nFull filtering, Retail analysis\n\nFewer permissive-check errors", size=23)
s.card(6.75, 1.95, 5.85, 3.7, "TRAINING ALGORITHM", "43.2% Pass@1\nOnline RL, Calculation\n\nMore instability and complexity", size=23)
s.text("Different experiments and denominators. Interpret each number in its own setting.", .77, 6.0, 11.9, .4, 18, MUTED)
s.source("Paper §5.3, Figures 3–4. Ambiguous tasks and limited cross-environment transfer remain limitations (§6).", PAPER)

s = add("Our implementation has two separate execution paths", "09 / Companion architecture", 65, """
The repository intentionally makes the educational baseline distinct from language-model execution. The CPU path samples among five hand-written strategies and updates their probabilities from rewards. The optional language-model path generates actual tool calls using a Hugging Face model and contains response-only LoRA training. Each run records which path produced its results. Within the language-model path, task generation is also explicit: templates provide an initial smoke configuration, while an optional interactive challenger can author tasks. Neither task source is silently substituted for the other. Tasks, conversations, selected training examples, metrics, and configuration are saved. This structure supports tracing a reported result back to the actual attempts and avoids presenting a lightweight demonstration as a reproduction of a large benchmark experiment.
""")
steps = [("Task source", "Template / optional LLM"), ("Quality filter", "Reference + failure probes"), ("Executor", "Tool observations only")]
for i, (label, body) in enumerate(steps):
    s.card(.65+4.2*i, 1.88, 3.95, 1.75, label, body, size=17)
s.card(.7, 4.03, 5.8, 1.9, "CPU PATH — COMPLETED", "Strategy probabilities updated\nMeasured on separate synthetic tasks", size=19)
s.card(6.75, 4.03, 5.85, 1.9, "LLM PATH — CHECK PENDING", "0/96 rollouts; then 0/3 probes\nNo adapter updates; native v3 pending", accent="A36C15", size=19)
s.source("Companion source: sca/agents.py, sca/llm.py, sca/cli.py. These are educational adaptations.", REPO)

s = add("Measure outcomes and keep evaluation separate", "10 / Metrics and protocol", 60, """
Pass at one and pass at four answer different questions in our experiment. The first uses the first sampled attempt for each test task. The second asks whether any of four attempts succeeded. A high pass-at-four score does not establish that the agent succeeds consistently on every attempt. TauBench separately introduced pass-hat-k for all-k-trials reliability, so the symbols should not be used interchangeably. The companion evaluator checks the requested goal and unchanged unrelated state. Its result is compared with the task verifier's reward, and disagreements are logged. Before and after evaluation uses the same test set and sampling settings. Training uses only training rollouts; neither test conversations nor reference solutions become hidden training examples. This protects the meaning of the before-and-after comparison.
""", (SOURCE_URLS["tau_bench"], SOURCE_URLS["cpu_results"]))
s.card(.7, 1.92, 3.78, 2.07, "PASS@1", "First attempt succeeds", size=23)
s.card(4.78, 1.92, 3.78, 2.07, "PASS@4", "Any of four succeeds", size=23)
s.card(8.86, 1.92, 3.78, 2.07, "COST + ERRORS", "Calls, errors, disagreements", size=21)
s.box(.7, 4.4, 11.93, 1.42, LIGHT)
s.text("Separate train / test tasks\nSame evaluation settings before and after; independent goal oracle", .95, 4.68, 11.4, .85, 22, TEAL, True)
s.source("Companion metric definitions; TauBench distinguishes pass@k from pass^k reliability. Links in notes.", SOURCE_URLS["tau_bench"])

s = add("Measured CPU results: useful, but narrowly scoped", "11 / Companion evidence", 65, """
This chart comes from the saved CPU example run. The setup contains one hundred and twenty training tasks, four attempts per training task, and sixty held-out test tasks. It is a categorical policy over five existing strategies. Successful training attempts increase the sampling weight of the strategy that generated them. The chart therefore shows learning to select a useful existing workflow more often; it does not show a language model inventing the workflow. Explain the pass-at-four value carefully: every test task had at least one success among its four updated-policy attempts, but some individual attempts still failed. Tool errors decreased, while mean tool calls increased because an effective workflow usually performs more work than stopping early. These results have one seed and a limited synthetic task family, so they do not establish broad generalization.
""", (SOURCE_URLS["cpu_results"],))
chart_y, chart_h = 2.1, 3.18
for ytick in (0, 50, 100):
    y = chart_y+chart_h*(1-ytick/100)
    s.line(1.0, y, 8.9, y, "DBE4EB", .8)
    s.text(str(ytick), .56, y-.09, .34, .23, 10, MUTED, align="right")
for i, key in enumerate(("pass_at_1", "pass_at_4")):
    center = 3.0+4.0*i
    for j, (stage, color) in enumerate((("before", BLUE), ("after", GREEN))):
        v = DATA[stage][key]*100
        x = center-.95+1.02*j
        bh = chart_h*v/100
        s.box(x, chart_y+chart_h-bh, .82, bh, color, radius=False)
        s.text(f"{v:.1f}%", x-.2, chart_y+chart_h-bh-.37, 1.22, .32, 17, color, True, align="center")
    s.text("Pass@1" if i==0 else "Pass@4", center-.78, 5.48, 2.0, .4, 21, INK, True, align="center")
s.box(9.4, 2.05, 3.15, 3.75, LIGHT)
s.text("120 train tasks\n60 test tasks\nSeed 42", 9.65, 2.33, 2.65, 1.55, 23, INK, True)
s.text("Before", 9.92, 4.37, 2, .35, 18, BLUE, True)
s.box(9.65, 4.4, .15, .18, BLUE, radius=False)
s.text("After", 9.92, 4.93, 2, .35, 18, GREEN, True)
s.box(9.65, 4.96, .15, .18, GREEN, radius=False)
s.source("Measured CPU strategy-policy run: results/cpu_demo/summary.json. Not an LLM or paper benchmark result.", SOURCE_URLS["cpu_results"])

s = add("Our filter ablation uses known fixture defects", "12 / Controlled experiment", 55, """
This experiment tests the filtering code with deliberate defects, not the natural error distribution of a language model. There are forty-eight candidate fixtures: twelve valid, twelve with bad reference solutions, twelve with permissive checks, and twelve with overstrict checks. Requiring only a valid schema keeps all of them. Requiring the reference solution to pass removes some defects, but can retain a verifier that accepts anything. Testing no-ops and failure cases addresses that particular weakness. In this constructed set, the final stage retains the twelve valid fixtures. That does not mean its precision will remain perfect on new tasks. The probe set is finite, and the accepted subsets differ between filters. Interpret the counts as a controlled code demonstration rather than an estimate of production task quality.
""", (SOURCE_URLS["cpu_results"],))
for i, (label, value) in enumerate(zip(("Schema only", "+ Reference check", "+ No-op / failures"), [r["accepted"] for r in ABLATIONS])):
    y = 2.13+i*1.08
    s.text(label, .8, y+.16, 3.75, .46, 22, INK, i==2)
    s.box(4.65, y+.08, 6.43, .57, LIGHT)
    s.box(4.65, y+.08, 6.43*value/48, .57, ["94A3B8", BLUE, GREEN][i])
    s.text(f"{value}/48", 11.35, y+.13, 1.1, .45, 22, TEAL, True)
s.text("12 valid + 12 bad solutions + 12 permissive + 12 overstrict", .8, 5.8, 11.8, .5, 21, MUTED)
s.source("Companion controlled fixtures: results/cpu_demo/filter_ablation.json. Not an LLM generation-quality estimate.", SOURCE_URLS["cpu_results"])

s = add("Two LLM attempts stopped before training", "13 / Observed failures and next check", 55, """
The first Colab run used Qwen2.5-0.5B-Instruct and generated ninety-six unsuccessful training attempts, so fine-tuning was skipped. The second used the 1.5B model and revised instructions, but none of its three readiness probes succeeded. That check stopped before full evaluation or training. The executed notebook preserved the summary and one probe transcript; it did not recover all three raw trajectories. The model still invented order IDs and skipped listing orders. The next configuration uses native function schemas, assistant tool-call messages, and tool-result roles. It also selects FP16 on the T4 instead of BF16 emulation. Neither change has a verified pretrained-model result yet. The next GPU attempt requests only a short readiness check. These are engineering changes, not evidence that model learning has improved performance.
""")
s.card(.7, 1.92, 5.8, 3.55, "OBSERVED ATTEMPTS", "First: 0.5B model\n0/96 rewarded rollouts\nSecond: 1.5B model\n0/3 readiness probes", size=22)
s.card(6.75, 1.92, 5.85, 3.55, "NEXT: NATIVE PROTOCOL", "Structured function schemas\nAssistant + tool-result roles\nT4 precision: FP16\nShort check only; no training", size=22)
s.text("STATUS: native-v3 model check pending. No LLM learning gain demonstrated.", .78, 5.98, 11.9, .42, 20, "A36C15", True)
s.source("Evidence: results/llm_first_attempt and results/llm_second_attempt. Notebook saves short-check traces inline.", REPO + "/tree/main/results/llm_second_attempt")

s = add("What the evidence can—and cannot—support", "14 / Limitations", 60, """
There are three boundaries I would keep visible when presenting the project. First, task tests cover examples, not every possible interpretation of an instruction. A reward can therefore be wrong even when the checker executes without error. Second, the companion CPU task family is intentionally narrow, and a single seed provides limited evidence. The useful next step is a larger independent test set and additional seeds, rather than a stronger claim about the same small run. Third, improvement must be assessed alongside resources. Rejecting many proposals, sampling more attempts, and making longer tool conversations can increase cost. My engineering interpretation is that the evaluator itself needs an audit. Record reward disagreements and inspect complete trajectories so that a score increase can be connected to intended behavior.
""")
s.card(.7, 1.92, 3.78, 3.8, "FEEDBACK", "Finite tests can miss errors.\n\nAudit reward vs. goal.", size=23)
s.card(4.78, 1.92, 3.78, 3.8, "GENERALIZATION", "One seed; narrow task family.\n\nUse independent tests.", size=23)
s.card(8.86, 1.92, 3.78, 3.8, "COST", "Generation, rejection, attempts.\n\nReport useful work per budget.", size=23)
s.text("Engineering interpretation: evaluate the feedback pipeline as carefully as the executor.", .75, 6.11, 11.9, .38, 18, TEAL, True)

s = add("Three takeaways for agent developers", "15 / Takeaways", 55, """
The first takeaway is to make practice material inspectable. Saving a task specification and its validation evidence makes it possible to diagnose bad feedback before an update is trusted. The second takeaway is to keep experimental identities clear. A categorical strategy policy, a pretrained language model, and a teacher-distillation setup answer different questions. A demonstration is useful when its claims match what it actually measures. The third takeaway is to preserve an independent evaluation boundary. Record the conditions and compare outcomes under the same test protocol, while tracking cost and disagreement. These choices make results easier to reproduce and critique. For the project, the next milestone is to run the pretrained-model experiment, then report what happened even if the measured score stays flat or decreases.
""", dark=True)
s.text("THREE TAKEAWAYS", .7, .55, 11.9, .4, 12, "8DDAD5", True)
for i, (a, b) in enumerate([("01", "Make practice tasks inspectable."), ("02", "Match claims to the experiment."), ("03", "Keep evaluation independent.")]):
    y = 1.65+1.25*i
    s.text(a, .8, y, 1.0, .55, 28, "8DDAD5", True)
    s.text(b, 2.05, y+.03, 10.4, .6, 29, WHITE, True)
s.text("Next: run the pretrained-model experiment and report the measured outcome.", .85, 6.03, 11.65, .58, 20, "DCE8EF")

s = add("Explore the article and reproduce the project", "16 / Resources", 35, """
These links connect the research explanation to the actual artifacts. The Medium article contains the narrative, the repository contains code and the measured CPU example, and the Colab notebook runs that code. The links are clickable in the PowerPoint and in the notes file. When recording, briefly show the repository structure or a saved trajectory if time allows. Save the executed notebook and run artifacts after the next experiment. Publishing slides and recording the presentation are separate assignment steps; generating this file does not publish it to SlideShare or create a video. The planned talk is about sixteen minutes, but verify the recorded duration before submission.
""", (MEDIUM, REPO, COLAB))
for i, (label, text, url) in enumerate([("MEDIUM ARTICLE", "Can AI Agents Learn by Creating Their Own Challenges?", MEDIUM), ("GITHUB REPOSITORY", "Deep-Dhaduk / Self-Challenging-Language-Model-Agents", REPO), ("COLAB NOTEBOOK", "Open the reproducible notebook", COLAB)]):
    y = 1.9+1.4*i
    s.box(.7, y, 11.9, 1.16, LIGHT)
    s.text(label, .95, y+.17, 10.9, .3, 12, TEAL, True)
    s.text(text, .95, y+.57, 11.0, .43, 21, INK, True, link=url)

s = add("References and evidence trail", "17 / Sources", 25, """
The primary case study, supporting survey, and related-method papers are listed here with direct links. The full source list also appears in the notes and the slide manifest. Consult the original papers for their experimental conditions. The CPU numbers come from saved repository artifacts, not from the research papers. This closes the evidence trail from source, to educational implementation, to reported result. Allow a short pause for questions or a final demonstration, and verify that the recorded presentation meets the assignment's minimum duration.
""", tuple(SOURCE_URLS.values()))
references = [("Zhou et al. (2025)", "Self-Challenging Language Model Agents", PAPER),
              ("Ren et al. (2026)", "Self-Improvements in Modern Agentic Systems", SURVEY),
              ("Wang et al. (2023)", "Self-Instruct", SOURCE_URLS["self_instruct"]),
              ("Yao et al. (2023)", "ReAct", SOURCE_URLS["react"]),
              ("Shinn et al. (2023)", "Reflexion", SOURCE_URLS["reflexion"]),
              ("Wang et al. (2024)", "Executable Code Actions / CodeAct", SOURCE_URLS["codeact"]),
              ("Yao et al. (2024)", "TauBench: tool-agent-user evaluation", SOURCE_URLS["tau_bench"]),
              ("Companion project", "Measured CPU artifacts and configuration", SOURCE_URLS["cpu_results"])]
for i, (author, title, url) in enumerate(references):
    x, y = .7+(i%2)*6.15, 1.85+(i//2)*1.15
    s.text(author, x, y, 5.85, .3, 14, TEAL, True)
    s.text(title, x, y+.35, 5.85, .6, 17, INK, link=url)


def render(slide, ppt_slide, index):
    image = Image.new("RGB", (round(W*DPI), round(H*DPI)), "white")
    draw = ImageDraw.Draw(image)
    for element in slide.elements:
        e = element
        x, y, w, h = [e[key] for key in ("x", "y", "w", "h")]
        if min(x, y) < -.001 or x+w > W+.01 or y+h > H+.01:
            raise ValueError(f"Off-slide element on slide {index}: {e}")
        xy = (round(x*DPI), round(y*DPI), round((x+w)*DPI), round((y+h)*DPI))
        if e["kind"] == "box":
            shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if e["radius"] else MSO_SHAPE.RECTANGLE
            shape = ppt_slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(e["fill"])
            if e["line"]:
                shape.line.color.rgb = RGBColor.from_string(e["line"])
            else:
                shape.line.fill.background()
            if e["radius"]:
                shape.adjustments[0] = .08
                draw.rounded_rectangle(xy, radius=11, fill="#"+e["fill"])
            else:
                draw.rectangle(xy, fill="#"+e["fill"])
        elif e["kind"] == "line":
            from pptx.enum.shapes import MSO_CONNECTOR
            shape = ppt_slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x+w), Inches(y+h))
            shape.line.color.rgb = RGBColor.from_string(e["color"])
            shape.line.width = Pt(e["width"])
            draw.line(xy, fill="#"+e["color"], width=max(1, round(e["width"]*DPI/72)))
        else:
            shape = ppt_slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tf = shape.text_frame
            tf.clear()
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.word_wrap = False
            tf.vertical_anchor = MSO_ANCHOR.TOP
            face = font(e["size"], e["bold"])
            for j, line in enumerate(e["text"].split("\n")):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[e["align"]]
                p.space_before = p.space_after = Pt(0)
                p.line_spacing = Pt(e["size"]*1.2)
                r = p.add_run()
                r.text = line
                r.font.name = "Arial"
                r.font.size = Pt(e["size"])
                r.font.bold = e["bold"]
                r.font.color.rgb = RGBColor.from_string(e["color"])
                if e["link"]:
                    r.hyperlink.address = e["link"]
                tx = x*DPI
                length = face.getlength(line)
                if e["align"] == "center":
                    tx += (w*DPI-length)/2
                elif e["align"] == "right":
                    tx += w*DPI-length
                draw.text((round(tx), round(y*DPI+j*e["size"]*1.2*DPI/72)), line, font=face,
                          fill="#"+e["color"], anchor="lt")
    notes = f"TARGET TIME: {slide.seconds} seconds\n\n{slide.notes}\n\nSOURCES\n" + "\n".join(slide.sources)
    ppt_slide.notes_slide.notes_text_frame.text = notes
    image.save(PREVIEWS / f"slide_{index:02d}.png")
    return image


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
prs.core_properties.title = "Can AI Agents Learn by Creating Their Own Challenges?"
prs.core_properties.subject = "Self-Challenging Language Model Agents: research case study and companion implementation"
prs.core_properties.author = "Deep Dhaduk"
prs.core_properties.keywords = "LLM agents, self-improvement, Code-as-Task, evaluation, individual assignment"
prs.core_properties.comments = "Paper findings, CPU results, and two unsuccessful LLM attempts are distinguished. Native-protocol pretrained-model check pending."
images = []
manifest = []
elapsed = 0
notes = ["PRESENTER NOTES\nCan AI Agents Learn by Creating Their Own Challenges?\n",
         "Timing is a rehearsal guide, not a guarantee of recorded duration.\n"]
for index, slide in enumerate(slides, 1):
    slide.footer(index, len(slides))
    images.append(render(slide, prs.slides.add_slide(prs.slide_layouts[6]), index))
    start, end = elapsed, elapsed+slide.seconds
    manifest.append({"slide": index, "title": slide.title, "section": slide.section,
                     "target_seconds": slide.seconds, "start_seconds": start, "end_seconds": end,
                     "speaker_note_words": len(slide.notes.split()), "sources": slide.sources})
    notes.append(f"\n{'='*72}\nSLIDE {index:02d}: {slide.title}\nTarget {start//60:02d}:{start%60:02d}–{end//60:02d}:{end%60:02d}\n\n{slide.notes}\n\nSources:\n" + "\n".join(slide.sources) + "\n")
    elapsed = end
destination = OUT / "self_challenging_agents.pptx"
prs.save(destination)
# Portable PDF visual edition, rendered from the same layout model as the PPTX.
images[0].save(OUT / "self_challenging_agents.pdf", save_all=True, append_images=images[1:], resolution=DPI)
(OUT / "speaker_notes.txt").write_text("\n".join(notes), encoding="utf-8")
(OUT / "slide_manifest.json").write_text(json.dumps({"slides": manifest, "total_target_seconds": elapsed,
    "speaker_note_words": sum(m["speaker_note_words"] for m in manifest), "sources": SOURCE_URLS,
    "pdf_note": "Visual edition rendered from the shared slide layout; PDF pages contain no speaker notes or hyperlinks."}, indent=2)+"\n", encoding="utf-8")

thumb_w, thumb_h = 480, 270
sheet = Image.new("RGB", (thumb_w*3, (thumb_h+34)*math.ceil(len(images)/3)), "#DFE7ED")
draw = ImageDraw.Draw(sheet)
for i, image in enumerate(images):
    x, y = (i%3)*thumb_w, (i//3)*(thumb_h+34)
    sheet.paste(image.resize((thumb_w, thumb_h)), (x, y))
    draw.text((x+10,y+thumb_h+5), f"{i+1:02d}  {slides[i].title[:49]}", fill="#142D43", font=font(8))
sheet.save(OUT / "slide_overview.png")

# Inspect the saved PowerPoint independently for slide count, notes, and links.
check = Presentation(destination)
assert len(check.slides) == 18
assert all(slide.notes_slide.notes_text_frame.text.strip() for slide in check.slides)
assert all(slide.shapes for slide in check.slides)
print(json.dumps({"pptx": str(destination), "slides": len(check.slides),
                  "target_duration": f"{elapsed//60}:{elapsed%60:02d}",
                  "speaker_note_words": sum(m["speaker_note_words"] for m in manifest),
                  "pdf": str(OUT / "self_challenging_agents.pdf")}, indent=2))
