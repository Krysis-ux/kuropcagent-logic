# architecture

> wip. numbers and rules below are what's in the code right now, they will move.

## big picture

```mermaid
flowchart TB
    subgraph client
        UI[web ui] 
        CLI[kuro cli]
        API[openai / anthropic compatible api]
    end

    subgraph server[kuro-server]
        CHAT[routes/chat.rs<br/>run_turn loop]
        RT[tools_runtime.rs<br/>assemble + dispatch]
        SUB[subagent.rs<br/>delegate]
    end

    subgraph core[kuro-core]
        ORC[orchestrate.rs<br/>effort -> plan]
        PR[prompt.rs<br/>system brief]
        INT[tools/intent.rs<br/>search or not]
        SK[skills]
        AG[agents]
        WS[workspace<br/>files + exec]
        MCP[mcp client]
        ENG[engine manager<br/>llama.cpp]
        FREE[free pool]
        GW[gateway<br/>anthropic <-> openai]
    end

    VIS[vision<br/>screen elements]

    UI & CLI & API --> CHAT
    CHAT --> INT
    CHAT --> ORC --> SK
    CHAT --> PR
    CHAT --> RT
    RT --> WS & MCP & SUB
    SUB --> AG
    CHAT --> ENG & FREE
    API --> GW
    VIS -. planned tool .-> RT
```

## one turn

```mermaid
sequenceDiagram
    participant U as user
    participant C as chat.rs
    participant I as intent
    participant O as orchestrate
    participant M as model
    participant T as tools

    U->>C: message + effort + mode
    C->>I: needs web?
    I-->>C: skip / search "query"
    C->>C: upfront search, read top 3 pages
    C->>O: plan(effort, surface, workspace, message)
    O-->>C: max rounds, skills, token budget
    C->>C: build system brief
    loop until no tool calls or rounds used up
        C->>M: messages + tools
        M-->>C: text and/or tool calls
        C->>T: dispatch each call
        T-->>C: result (appended to messages)
    end
    C-->>U: stream reply + sources + stats
```

on the last round tools are not offered, so the model has to answer.

## orchestration

`kuro-core/src/orchestrate.rs`

input: effort, surface (chat or code), workspace root + mode, the message,
pinned skills. output: a `Plan`.

| effort | chat rounds | code rounds | chat budget | code budget |
|---|---:|---:|---:|---:|
| instant | 2 | 4 | 200 | 400 |
| balanced | 4 | 8 | 500 | 1000 |
| thinking | 6 | 12 | 900 | 1800 |
| extended | 8 | 16 | 1400 | 3000 |
| ultra | 8 | 40 | 1400 | 8000 |

budget = tokens of skill text allowed in the system prompt.

how skills get picked:

1. essentials always on when coding (read before edit, run the tests, ...)
2. languages from project files: `Cargo.toml` -> rust, `package.json` ->
   typescript (+ react / tailwind from deps), `pyproject.toml` -> python, etc.
   max 3.
3. keyword triggers in the message (`slow` -> performance, `why` -> debugging)
4. everything ranked, then trimmed to fit the budget. skills named in the
   message survive any trim. what got dropped is reported back.

if auto orchestration is off, it just uses the user's skills as-is and only
budgets rounds.

## subagents

`kuro-core/src/agents/mod.rs`, `kuro-server/src/routes/subagent.rs`

same model, different brief + skills, fresh context. one per kind of work:

| agent | writes | for |
|---|:---:|---|
| explore | no | figure out how something works |
| frontend | yes | ui code |
| backend | yes | handlers, schemas, services |
| debug | yes | find cause of a failure |
| test | yes | write + run tests |
| review | no | review a change |
| architect | no | design before building |
| refactor | yes | reshape without behavior change |

non-writing agents only get read tools. a subagent never gets more than the
parent workspace mode allows.

```mermaid
flowchart LR
    MAIN[main turn] -->|delegate explore 'why does login fail'| SA[subagent loop<br/>fresh context]
    SA --> RD[read / grep / run]
    RD --> SA
    SA -->|1 paragraph| MAIN
```

## workspace modes

| mode | can do |
|---|---|
| ask | nothing, can't see project |
| plan | read + search |
| agent | read, write, run dev commands in folder |
| bypass | agent without the command allowlist |

paths get resolved (symlinks, `..`) before the root check. `.ssh`, `.aws`,
`.env`, private keys are refused. every write keeps the old contents for undo.

## models

- local: one llama.cpp process per loaded model, supervised by `engine/manager.rs`
- providers: bring your own key (openrouter, anthropic, openai, groq, ...)
- kuro free: pool of free-tier keys, each request goes to whichever provider
  still has allowance, cooldown on 429 / quota errors
- `classify.rs` reads model ids to filter out embedders/rerankers/etc so they
  don't show up as chat models

## where the pc agent plugs in

the screen loop (see [vision.md](vision.md)) is meant to be just another tool
group in `tools_runtime.rs`: `screen_look` returns numbered elements,
`screen_act` clicks/types by element id. that way it gets the same round
limits, effort dial and mode gating as everything else. not wired yet.
