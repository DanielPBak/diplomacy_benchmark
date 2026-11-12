"""Stand-in for `claude -p` in tests: echoes a schema-shaped reply and records its inputs."""
import json
import os
import sys

args = sys.argv[1:]
raw = sys.stdin.read()
if "--input-format" in args and args[args.index("--input-format") + 1] == "stream-json":
    blocks = [b["text"] for b in json.loads(raw.splitlines()[0])["message"]["content"]]
    prompt = "\n\n".join(blocks)
else:
    blocks, prompt = [raw], raw
log = os.environ.get("FAKE_CLAUDE_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"args": args, "prompt": prompt, "blocks": len(blocks), "cwd": os.getcwd(),
                            "max_out": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")}) + "\n")
if os.environ.get("FAKE_CLAUDE_FAIL"):
    print(json.dumps({"type": "result", "is_error": True, "api_error_status": 429, "result": "rate limited",
                      "subtype": "error"}))
    sys.exit(1)
schema = json.loads(args[args.index("--json-schema") + 1])
out = {k: ([] if v.get("type") == "array" else False if v.get("type") == "boolean" else "")
       for k, v in schema["properties"].items()}
if "orders" in out and "YOUR ORDERABLE UNITS AND LEGAL ORDERS THIS PHASE:\n" in prompt:
    block = prompt.rsplit("YOUR ORDERABLE UNITS AND LEGAL ORDERS THIS PHASE:\n", 1)[1].split("\n\n", 1)[0]
    out["orders"] = [l.split(": ", 1)[1].split(" | ")[0] for l in block.splitlines() if ": " in l]
print(json.dumps({"type": "system", "subtype": "init"}))
print(json.dumps({"type": "result", "is_error": False, "subtype": "success", "result": json.dumps(out), "structured_output": out,
                  "total_cost_usd": 0.0123, "usage": {"input_tokens": 5, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 50, "output_tokens": 20,
                  "output_tokens_details": {"thinking_tokens": 7}}}))
