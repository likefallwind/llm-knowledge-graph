from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from kg.embeddings import cosine_scores
from kg.llm import LLMConfig, MiniMaxM3LLM

config = replace(LLMConfig.from_env(), model="MiniMax-M3", base_url="https://api.minimaxi.com/v1", timeout=180, retries=3)
try:
    scores = cosine_scores("reinforcement learning", ["Reinforcement learning studies learning from reward."])
    assert len(scores) == 1
    response = MiniMaxM3LLM(config).complete_json("Return only a JSON object.", 'Return {"ok": true}.')
    assert response.get("ok") is True
    report = dict(at=datetime.now(timezone.utc).isoformat(), embedding_ok=True, api_ok=True, model=config.model, endpoint=config.endpoint)
    Path(__file__).with_name("preflight-runtime.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
except Exception as exc:
    print(type(exc).__name__ + ": " + str(exc).replace(config.api_key, "[REDACTED]"), flush=True)
    raise SystemExit(1)
