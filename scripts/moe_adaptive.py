#!/usr/bin/env python3
"""MoE standalone tuning and request-level adaptation for DFlash and MTP."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def controls(cpu_cap, gpu):
    return {
        "CUDA_VISIBLE_DEVICES": gpu,
        "GGML_MOE_HYBRID": "1",
        "GGML_MOE_HYBRID_POLICY": "legacy",
        "GGML_MOE_HYBRID_MIN_TOKENS": "9",
        "GGML_MOE_HYBRID_MAX_TOKENS": "8",
        "GGML_MOE_HYBRID_MAX_EXPERTS": str(cpu_cap),
        "GGML_MOE_HYBRID_REUSE_WINDOW": "0",
        "GGML_MOE_HYBRID_THREADS": "8",
        "GGML_MOE_PREFETCH_BUDGET": "-1",
        "GGML_MOE_PREFETCH_RANK": "votes",
        "GGML_MOE_PREFETCH_PROTECT": "1",
        "GGML_MOE_CACHE_POLICY": "lfu",
        "GGML_SCHED_MOE_SLOT_STATS": "0",
        "GGML_SCHED_MOE_KREL": "0",
    }


def environment(overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "DFLASH_"))}
    env.update(overrides)
    return env


def http(port, endpoint, payload=None, timeout=1200):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=body,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def require_idle_gpu(gpu):
    used = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    if int(used) > 512:
        raise RuntimeError(f"Selected GPU uses {used} MiB; other container processes may be hidden")
    active = subprocess.check_output(["nvidia-smi", "-i", gpu,
                                      "--query-compute-apps=pid,process_name,used_gpu_memory",
                                      "--format=csv,noheader"], text=True, timeout=15).strip()
    if active:
        raise RuntimeError(f"Selected GPU has running compute processes; retry when idle:\n{active}")


def launch(args, port):
    profile = args.profile.resolve() if args.profile else args.output / "costs.json"
    spec_type = getattr(args, "spec_type", "draft-dflash")
    draft_max = getattr(args, "draft_max", 7)
    key = {"target": file_identity(args.model), "draft": file_identity(args.draft_model or args.model),
           "spec_type": spec_type, "draft_max": draft_max,
           "binary": file_identity(args.server),
           "runtime_libraries": {p.name: file_identity(p.resolve()) for p in sorted(args.server.parent.glob("*.so*")) if p.is_file()},
           "gpu": args.gpu, "threads": args.threads,
           "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    env = controls(args.cpu_cap if args.action == "serve" else 0, args.gpu)
    env.update({"LLAMA_MOE_AUTOTUNE": "online" if args.action == "serve" else "offline",
                "LLAMA_MOE_AUTOTUNE_OUTPUT": str(args.output / "telemetry.jsonl"),
                "LLAMA_MOE_AUTOTUNE_PROFILE": str(profile),
                "LLAMA_MOE_AUTOTUNE_PROFILE_KEY": digest(key),
                "LLAMA_MOE_AUTOTUNE_WINDOW": str(args.window),
                "LLAMA_MOE_AUTOTUNE_SAMPLE_TOKENS": str(getattr(args, "sample_tokens", 20)),
                "LLAMA_MOE_AUTOTUNE_ADAPT": str(int((args.action == "serve" and not getattr(args, "observe_only", False)) or getattr(args, "validate_candidates", False))),
                "LLAMA_MOE_AUTOTUNE_COST_ROUNDS": str(0 if args.action == "serve" else args.cost_rounds),
                "LLAMA_MOE_AUTOTUNE_TRIAL_BUDGET": str(getattr(args, "trial_budget", 24)),
                "LLAMA_MOE_AUTOTUNE_WARMUP_REQUESTS": str(int(getattr(args, "warmup_tokens", 0) > 0)),
                "LLAMA_MOE_AUTOTUNE_CONFIRM_REQUESTS": str(int(getattr(args, "confirm_tokens", 1024) > getattr(args, "tokens", 1024))),
                "LLAMA_MOE_AUTOTUNE_NPRED": str(args.npred),
                "LLAMA_MOE_AUTOTUNE_THRESHOLD": str(args.threshold),
                "LLAMA_MOE_AUTOTUNE_CPU_CAP": str(args.cpu_cap),
                "LLAMA_MOE_AUTOTUNE_MAX_PRED": "24",
                "LLAMA_MOE_AUTOTUNE_DETAIL": str(int(args.detail or args.action != "serve"))})
    command = [str(args.server), "-m", str(args.model),
               "--spec-type", spec_type, "-ngl", "99", "-ngld", "99", "-c", str(args.ctx),
               "-b", "512", "-ub", "512", "-np", "1", "-t", str(args.threads), "-tb", str(args.threads),
               "--host", "127.0.0.1", "--port", str(port), "--fit", "off", "--load-mode", "none",
               "--no-warmup", "--temp", "0", "--ctx-checkpoints", "0", "--reasoning", "off", "--moe-substitute-threshold", "0",
               "--n-cache-layers", str(args.cache_layers), "--n-cache-slots", str(args.slots),
               "--n-cache-predict", "24", "--n-cache-pin", "32768",
               "--spec-draft-n-max", str(draft_max), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
               "--spec-draft-threads", str(args.threads), "--spec-draft-threads-batch", str(args.threads)]
    if args.draft_model:
        command += ["-md", str(args.draft_model)]
    return command, environment(env), {"identity": key, "command": command, "environment": env}


def read_events(path, summaries_only=False):
    events = []
    with path.open() as file:
        for line in file:
            if not line.endswith("\n"):
                break
            if summaries_only and line.startswith(('{"event":"round",', '{"event":"route",')):
                continue
            if line.strip():
                events.append(json.loads(line))
    return events


def search(tool, trace, output, request=None):
    command = [str(tool.resolve(strict=True)), str(trace.resolve(strict=True)), str(output.resolve())]
    if request is not None:
        command.append(str(request))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(output.read_text())
    print("Predicted recommendation:", json.dumps(result["selected"]),
          f"supported candidates={result['supported_candidates']}")
    print(f"Search compute={result['search_ms']:.3f} ms; load and search={result['load_and_search_ms']:.3f} ms")
    print(f"Full ranking: {output}")
    return result


def report(root, print_rounds=True):
    events = read_events(root / "telemetry.jsonl")
    rounds = [r for r in events if r["event"] == "round"]
    routes = [r for r in events if r["event"] == "route"]
    final = next((r for r in reversed(events) if r["event"] in ("final", "request_end", "window")), None)
    result_path = root / "search.json"
    numerical_search = json.loads(result_path.read_text()) if result_path.exists() else None
    recommendation = next((r for r in reversed(events) if r["event"] == "recommendation"), numerical_search)
    if not final:
        raise ValueError("No completed calibration report; inspect server.log")
    requests = {r["request"]: r for r in events if r["event"] in ("final", "request_end", "window")}
    summary = {"round_count": sum(r["rounds"] for r in requests.values()), "sampled_round_count": len(rounds), "traced_round_count": len(rounds) + len(routes),
               "full_rounds": sum(r["full"] for r in rounds + routes),
               "cost_probe_rounds": sum(r["cost_probe"] for r in rounds),
               "right_censored_rounds": sum(r["right_censored"] for r in rounds + routes),
               "final": final, "recommendation": recommendation, "numerical_search": numerical_search,
               "online_recommendations": [e for e in events if e["event"] == "recommendation"]}
    if (root / "session.json").exists():
        summary["session"] = json.loads((root / "session.json").read_text())
    write_json(root / "summary.json", summary)
    print(f"\nObserved {summary['sampled_round_count']} of {summary['round_count']} decode rounds (times in ms):")
    print(f"Cache replay records={summary['traced_round_count']}; service samples={summary['sampled_round_count']}")
    if print_rounds:
        for r in rounds:
            confidence = ",".join(f"{p:.5f}" for p in r["confidence"])
            print(f"r={r['round']:3} conf=[{confidence}] kept={r['kept']} accepted={r['accepted']} "
                  f"censored={r['right_censored']} draft={r['draft_ms']:.3f} verify={r['verify_ms']:.3f} "
                  f"miss_copies={r['demand_copies']} prefetch_copies={r['prefetch_copies']} "
                  f"cpu_assignments={r['cpu_assignments']} cost_probe={r['cost_probe']}")
    print("\nMeasured service costs:")
    for b, value in enumerate(final["costs"]["base"], 1):
        if value["samples"]:
            print(f"verify residual B={b}: {value['us']:.3f} us, samples={value['samples']}")
    for b, value in enumerate(final["costs"].get("cpu_residual", []), 1):
        if value["samples"]:
            active = final["costs"]["cpu_layers"][b - 1]["us"]
            print(f"CPU verify residual B={b}: {value['us']:.3f} us, samples={value['samples']}, active layers={active:.2f}")
    for phase, table in enumerate(final["costs"].get("phases", []), 1):
        for b, value in enumerate(table["base"], 1):
            cpu = table["cpu_residual"][b - 1]
            if value["samples"] or cpu["samples"]:
                print(f"Residual B={b} shape_run={phase}: GPU={value['us']:.3f} us n={value['samples']}, CPU={cpu['us']:.3f} us n={cpu['samples']}")
    for shape, costs in final["costs"]["shapes"].items():
        print(f"shape={shape}")
        for name in ("transfer", "issue", "cpu_per_assignment", "protocol"):
            value = costs[name]
            print(f"  {name}: {value['us']:.3f} us, samples={value['samples']}")
        for b, value in enumerate(costs["gpu"], 1):
            if value["samples"]:
                print(f"  GPU B={b}: {value['us']:.3f} us, samples={value['samples']}")
        for item in costs["joint"]:
            value = item["time"]
            print(f"  joint B={item['batch']} assignments={item['assignments']}: {value['us']:.3f} us, samples={value['samples']}")
    print(f"\nCost probe rounds={summary['cost_probe_rounds']}, probe execution={sum(r['cost_probe_ms'] for r in requests.values()):.3f} ms")
    print(f"Cost warmup rounds={sum(r.get('cost_warmup_rounds', 0) for r in requests.values())}, warmup execution={sum(r.get('cost_warmup_ms', 0) for r in requests.values()):.3f} ms")
    print(f"Controller={final['controller_ms']:.3f} ms; native collection elapsed={final['elapsed_ms']:.3f} ms")
    if "session" in summary:
        for name, value in summary["session"].get("elapsed_seconds", {}).items():
            print(f"{name}={value:.3f} s")
    if recommendation:
        print("Recommended:", json.dumps(recommendation["selected"]),
              "status=" + recommendation.get("status", "predicted"))
    print(f"Full values: {root / 'telemetry.jsonl'}")
    return summary


def run(args):
    args.output = args.output.resolve()
    for name in ("server", "model"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.draft_model:
        args.draft_model = args.draft_model.resolve(strict=True)
    if args.spec_type == "draft-dflash" and not args.draft_model:
        raise ValueError("DFlash requires --draft-model")
    if not 1 <= args.draft_max <= 7:
        raise ValueError("Draft maximum must be between 1 and 7")
    if args.action != "serve":
        if not args.prompt:
            raise ValueError("tune requires --prompt")
    if args.tokens is None:
        args.tokens = 1024 if args.spec_type == "draft-mtp" else 512
    if args.warmup_tokens is None:
        args.warmup_tokens = 256 if args.spec_type == "draft-mtp" else 0
    if args.confirm_tokens is None:
        args.confirm_tokens = max(1024, args.tokens)
    if not 0 <= args.npred <= 24 or not 0 <= args.threshold <= 1 or not 0 <= args.cpu_cap <= 8:
        raise ValueError("Invalid fixed inference configuration")
    if not 10 <= args.trial_budget <= 24 or (args.warmup_tokens != 0 and not 65 <= args.warmup_tokens <= 32768):
        raise ValueError("Invalid candidate trial budget or warmup length")
    if not args.tokens <= args.confirm_tokens <= 32768 or args.tokens < 65 or args.tokens > 32768 or args.ctx < 512 or args.threads < 1:
        raise ValueError("Invalid token/context/thread count")
    if not 16 <= args.window <= 128 or not 0 <= args.cost_rounds <= 96 or args.slots < 1:
        raise ValueError("Invalid window, probe budget, or fixed slots")
    if args.action == "serve" and not 1 <= args.sample_tokens <= 10000:
        raise ValueError("Sample interval must be between 1 and 10000 output tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.profile:
        args.profile = args.profile.resolve()
        if not args.profile.parent.is_dir():
            raise ValueError("Profile parent directory must exist")
    require_idle_gpu(args.gpu)
    port = args.port
    if not port:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    args.validate_candidates = args.action == "tune"
    command, env, manifest = launch(args, port)
    write_json(args.output / "manifest.json", manifest)
    if args.action == "serve":
        print(f"Online {args.spec_type} (collect during generation; compare candidates after requests): http://127.0.0.1:{port}", flush=True)
        print(f"Telemetry: {args.output / 'telemetry.jsonl'}", flush=True)
        os.execve(str(args.server), command, env)
    started = time.perf_counter()
    session = {"status": "running", "elapsed_seconds": {}, "candidate_trials": []}
    process = None
    try:
        with (args.output / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited; inspect server.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server startup timed out")
                try:
                    if http(port, "/health", timeout=2).get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            session["elapsed_seconds"]["load_and_startup"] = time.perf_counter() - started
            prompt = args.prompt.read_text()
            if not args.raw_prompt:
                prompt = http(port, "/apply-template", {"messages": [{"role": "user", "content": prompt}]})["prompt"]
            tokens = http(port, "/tokenize", {"content": prompt, "add_special": True})["tokens"]
            if len(tokens) + max(args.confirm_tokens, args.warmup_tokens) + args.draft_max + 1 > args.ctx:
                raise ValueError("Prompt plus output exceeds context")
            session["input_tokens"] = len(tokens)
            session["requested_output_tokens"] = args.tokens
            request = {"prompt": prompt, "n_predict": args.tokens, "temperature": 0, "seed": 20261027,
                       "cache_prompt": False, "return_tokens": True, "stream": False,
                       "ignore_eos": args.ignore_eos}
            write_json(args.output / "request.json", request)
            request_start = time.perf_counter()
            response = http(port, "/completion", request, timeout=args.timeout)
            session["elapsed_seconds"]["calibration_request"] = time.perf_counter() - request_start
            write_json(args.output / "response.json", response)
            session["timings"] = response.get("timings", {})
            session["elapsed_seconds"]["prefill"] = session["timings"].get("prompt_ms", 0) / 1000
            session["elapsed_seconds"]["decode_including_collection"] = session["timings"].get("predicted_ms", 0) / 1000
            session["actual_output_tokens"] = len(response.get("tokens", []))
            comparison_start = time.perf_counter()
            recommendation = None
            for request_id in range(1, 2 * args.trial_budget + 2):
                deadline = time.monotonic() + args.timeout
                while True:
                    events = read_events(args.output / "telemetry.jsonl", summaries_only=True)
                    if any(e["event"] == "request_end" and e["request"] == request_id for e in events):
                        break
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Candidate comparison did not complete; inspect server.log")
                    time.sleep(0.02)
                recommendation = next((e for e in reversed(events) if e["event"] == "recommendation"), None)
                if not recommendation or recommendation["status"] != "candidate_trials" or request_id == 2 * args.trial_budget + 1:
                    break
                trial_started = time.perf_counter()
                trial_role = recommendation.get("next_trial_role", "measurement")
                trial_request = dict(request, n_predict=args.warmup_tokens if trial_role == "warmup" else args.confirm_tokens if trial_role == "confirmation" else args.tokens)
                trial_response = http(port, "/completion", trial_request, timeout=args.timeout)
                write_json(args.output / f"candidate-response-{request_id}.json", trial_response)
                session["candidate_trials"].append({"config": recommendation["next_config"], "role": trial_role,
                    "requested_tokens": trial_request["n_predict"],
                    "elapsed_seconds": time.perf_counter() - trial_started, "timings": trial_response.get("timings", {})})
            session["elapsed_seconds"]["candidate_comparison"] = time.perf_counter() - comparison_start
            session["native_recommendation"] = recommendation
            session["recommended"] = recommendation["selected"] if recommendation and recommendation["status"] == "validated" else None
            session["selection_status"] = recommendation["status"] if recommendation else "no_recommendation"
            session["status"] = "collected"
    except BaseException as error:
        session.update(status="failed", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_start = time.perf_counter()
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            session["elapsed_seconds"]["shutdown"] = time.perf_counter() - stop_start
        session["elapsed_seconds"]["collection_total"] = time.perf_counter() - started
        session["elapsed_seconds"]["total"] = session["elapsed_seconds"]["collection_total"]
        session["exit_code"] = process.returncode if process else None
        if session["status"] == "collected" and session["exit_code"] != 0:
            session.update(status="failed", error="Server shutdown failed; inspect server.log")
        write_json(args.output / "session.json", session)
    if session["exit_code"] != 0:
        raise RuntimeError("Server shutdown failed; inspect server.log")
    search_started = time.perf_counter()
    try:
        result = next((dict(e) for e in events if e["event"] == "recommendation" and e["request"] == 1 and "ranking" in e), None)
        if result is None:
            raise ValueError("Insufficient observations for initial numerical search")
        result["selected"] = result.get("predicted_selected")
        write_json(args.output / "search.json", result)
        session["elapsed_seconds"]["native_search_total"] = sum(e["search_ms"] for e in events if e["event"] == "recommendation") / 1000
        print("Native recommendation:", json.dumps(session["recommended"]), session["selection_status"])
        session["elapsed_seconds"]["search_compute"] = result["search_ms"] / 1000
        session["status"] = "complete" if session["selection_status"] == "validated" else "incomplete"
    except BaseException as error:
        session.update(status="failed", error=str(error))
        raise
    finally:
        session["elapsed_seconds"]["search"] = time.perf_counter() - search_started
        session["elapsed_seconds"]["total"] = time.perf_counter() - started
        write_json(args.output / "session.json", session)
    report(args.output)
    print(f"Standalone tuning {session['status']}. Observations and selection status are saved in", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "serve"):
        p = sub.add_parser(action, aliases=["calibrate"] if action == "tune" else [],
                           help="Collect calibration data and search in one call" if action == "tune" else "Collect real requests and compare retained configurations after responses")
        p.set_defaults(action=action)
        p.add_argument("--server", type=Path, required=True)
        if action == "tune":
            p.add_argument("--tool", type=Path, help="Compatibility option; tune uses the native server search")
        else:
            p.add_argument("--sample-tokens", type=int, default=20, help="Collect after each N output tokens; default 20, plus the first decode step")
            p.add_argument("--observe-only", action="store_true", help="Collect without selecting a configuration for the next request")
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--spec-type", choices=("draft-dflash", "draft-mtp"), default="draft-dflash")
        p.add_argument("--draft-max", type=int, default=7, help="Draft limit per session: 1..7, also bounded by the draft model's capacity")
        p.add_argument("--draft-model", type=Path, help="Required for DFlash; MTP defaults to the heads embedded in --model")
        p.add_argument("--gpu", required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--profile", type=Path, help="Read/update a compatible cost profile; default OUTPUT/costs.json")
        p.add_argument("--prompt", type=Path)
        p.add_argument("--tokens", type=int, help="Initial collection and candidate output: default 1024 for MTP, 512 for DFlash")
        p.add_argument("--confirm-tokens", type=int, help="Confirmation output length; defaults to max(1024, --tokens)")
        p.add_argument("--warmup-tokens", type=int, help="Candidate warmup output; defaults to 256 for MTP, zero for DFlash")
        p.add_argument("--trial-budget", type=int, default=24, help="Maximum screening and confirmation requests combined")
        p.add_argument("--ctx", type=int, default=8192)
        p.add_argument("--slots", type=int, default=64)
        p.add_argument("--cache-layers", type=int, default=4)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--npred", type=int, default=0, help="Initial online prediction prefix width")
        p.add_argument("--threshold", type=float, default=0, help="Initial online draft threshold")
        p.add_argument("--cpu-cap", type=int, default=0, help="Initial online CPU expert cap")
        p.add_argument("--window", type=int, default=32, help="Collected timing samples per report; search uses the whole request")
        p.add_argument("--cost-rounds", type=int, default=48, help="Measured cost probes after one unmeasured warmup pass; zero disables both")
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--port", type=int, default=8080 if action == "serve" else 0)
        p.add_argument("--raw-prompt", action="store_true")
        p.add_argument("--detail", action="store_true", help="Write full per-layer records on collected steps")
        p.add_argument("--ignore-eos", action="store_true", help="Fixed-length benchmark only; normal calibration respects EOS")
    p = sub.add_parser("search", help="Rank saved observations without running inference or changing a server")
    p.add_argument("--tool", type=Path, required=True, help="Path to llama-moe-autotune")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--request", type=int, help="Request ID; defaults to the last request")
    p = sub.add_parser("report")
    p.add_argument("output", type=Path)
    p.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "search":
            search(args.tool, args.trace, args.output, args.request)
        elif args.action == "report":
            report(args.output, not args.summary_only)
        else:
            run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"Adaptive tuning failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
