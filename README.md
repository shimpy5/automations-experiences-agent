# Automations Experiences Agent

Name a smart-home experience in plain language — *"make it cozy for movie
night"* — and an **agent** sets it up in your real home: it looks up a matching
pattern, reads your actual devices, builds the scene, runs it, and when a device
doesn't respond it decides on its own whether to retry, substitute, or skip.

This is the agent counterpart to a grounded RAG pipeline —
[smartthings-experience-engine](https://github.com/shimpy5/smartthings-experience-engine).
The RAG version makes
**one** model call that returns a finished scene, which code then executes. Here
the model runs a **tool loop**: it chooses which tool to call and in what order,
sees each result, and reacts — including to failures. The model supplies the
judgment; the code supplies the guardrails.

Runs end to end against a simulated home — no hardware, no account.

## How it works

The model is given five tools and a loop drives them:

```
request ─▶ get_experience ─▶ get_home ─▶ apply_scene ─▶ (failure?) ─▶ finish
            (recipe pattern)  (real        (run the        │
                               devices)     whole scene)    └▶ execute_command
                                                               (retry / substitute / skip)
```

- **`get_experience`** — match the request to an abstract recipe pattern (dim
  the Living Room lights, warm, shades down), naming no specific device.
- **`get_home`** — the real devices, their capabilities, and current state. The
  model binds the pattern to *these* devices.
- **`apply_scene`** — run the whole scene at once (the fast, happy path). It
  validates every command, applies what it can, and reports which devices
  succeeded, failed, or were invalid. It does **not** roll back successful
  commands when another fails.
- **`execute_command`** — run a single device, for when the model wants to retry
  a failed one, use a different device, or make a fix.
- **`finish`** — end with a short summary.

This is a deliberate **hybrid** design: the scene runs in one bulk call on the
happy path, and the model only drops to per-device control when something fails.
That keeps the cost down while still letting the model handle failures itself.

### What lives in code vs. the model

The model decides the scene (which devices, what values), the order, and how to
handle a failure. Two things it is never trusted with stay in code:

- **Validation** ([`smarthome/validation.py`](smarthome/validation.py)) — every
  command is checked against a capability schema before it runs. A made-up
  device or an out-of-range value is rejected with a reason (which the model
  gets back and can fix), never sent to a device.
- **Restore backstop** ([`agent/tools.py`](agent/tools.py)) — each device's
  starting state is remembered the first time it's touched. If a run aborts (hits
  the step limit or crashes), the loop puts the home back, so a half-applied room
  is never left stranded. It is not a tool the model controls.

## Run it

No hardware needed — it runs against a simulated home.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # the agent calls the model

python3 -m agent.loop "make it cozy for movie night"
```

Watch it handle a dead device by forcing one to fail:

```bash
FAIL_DEVICES=lr-tv python3 -m agent.loop "set up movie night"
```

The model is `claude-opus-5-5` by default; override with `ANTHROPIC_MODEL`.

## Eval

A small, deterministic eval scores the agent so changes can be compared with a
number instead of a feeling ([`evals/`](evals)):

```bash
python3 -m evals.run_eval                 # all cases
python3 -m evals.run_eval movie_night     # one case
```

Each case runs the real model against a fresh simulated home, then scores the
**final home state** with range/intent checks (`dimmed` = level ≤ 30, not an
exact value — the model picks the number). Cases include a dead-device run that
checks the agent finished with a partial scene and didn't strand the room. The
simulated home makes scoring free and repeatable even though each case calls the
model, so keep the set small.

## Layout

```
smarthome/     devices, capability schema, state rules, mock home (+ fault
               injection), the validation guardrail
experiences/   abstract experience patterns + keyword retrieval
agent/
  tools.py     the five tools (definitions + handlers) + restore backstop
  loop.py      the hand-written tool loop that drives the model
evals/         test cases + a deterministic scorer
```

The only dependency is the `anthropic` SDK; everything else is plain Python. The
code is synchronous and the home is in-memory, to keep the agent easy to read
and run while learning. The clean seam at `smarthome/` is where a real
SmartThings backend would slot in.

## See also

- **[smartthings-experience-engine](https://github.com/shimpy5/smartthings-experience-engine)**
  — the same problem solved as a grounded RAG pipeline: one model call returns a
  validated scene that code executes, with a full experience lifecycle (enter,
  event-driven maintain, restore). Same idea, different architecture — one model
  call vs. the model driving a tool loop.
