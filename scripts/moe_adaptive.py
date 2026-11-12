#!/usr/bin/env python3
"""MoE standalone tuning and request-level adaptation for DFlash and MTP."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cpu_affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    if os.name == "nt":
        import ctypes
        mask, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(mask), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [cpu for cpu in range(ctypes.sizeof(mask) * 8) if mask.value & (1 << cpu)]
    return None


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


def replace_option(command, option, value):
    result = list(command)
    result[result.index(option) + 1] = str(value)
    return result


def apply_cpu_set(cpu_set, threads):
    if cpu_set == "none":
        return
    selected = set()
    for part in cpu_set.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError("Invalid production CPUSET")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first or last >= 64:
            raise ValueError("Invalid production CPUSET range")
        selected.update(range(first, last + 1))
    if len(selected) < threads or not selected.issubset(cpu_affinity() or []):
        raise ValueError("Production CPUSET is unavailable or smaller than the thread budget")
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        mask = sum(1 << cpu for cpu in selected)
        if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, selected)


def deployment(args):
    # The production launcher is the single source of inference settings.
    overrides = dict(os.environ)
    if args.gpu is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.thinking is not None:
        overrides["THINKING"] = args.thinking
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                             "-File", str(args.launcher), "--export-config"],
                            capture_output=True, text=True, encoding="utf-8", env=overrides, check=True)
    config = json.loads(result.stdout.lstrip("\ufeff"))
    if config.get("version") != 1:
        raise ValueError("Unsupported production configuration")
    if any(k.startswith("LLAMA_MOE_AUTOTUNE") for k in config["environment"]):
        raise ValueError("Remove LLAMA_MOE_AUTOTUNE variables before exporting a production configuration")
    command = config["command"]
    def value(flag):
        return command[command.index(flag) + 1]
    args.server = Path(command[0]).resolve(strict=True)
    args.model = Path(value("-m")).resolve(strict=True)
    args.spec_type = value("--spec-type")
    args.draft_model = Path(value("--spec-draft-model")) if "--spec-draft-model" in command else None
    args.draft_max = int(value("--spec-draft-n-max"))
    args.ctx = int(value("-c"))
    args.thinking = value("--reasoning")
    args.gpu = config["environment"]["CUDA_VISIBLE_DEVICES"]
    args.npred = int(value("--n-cache-predict"))
    args.prefetch_budget = config["selected"]["prefetch_budget"]
    args.threshold = config["selected"]["threshold"]
    args.cpu_cap = config["selected"]["cpu_cap"]
    args.threads = config["main_threads"]
    args.expert_threads = config["expert_threads"]
    args.slots = int(value("--n-cache-slots"))
    if args.spec_type not in ("draft-mtp", "draft-dflash"):
        raise ValueError("Tuning requires MTP or DFlash")
    if config["environment"]["GGML_MOE_HYBRID_POLICY"] != "legacy":
        raise ValueError("Tuning currently supports the production legacy CPU policy")
    return config


def production_command(args, port, config, log_path):
    command = args.deployment["command"]
    for flag, value in (("--host", "127.0.0.1"), ("--port", port), ("--log-file", log_path),
                        ("--spec-draft-p-min", config["threshold"])):
        command = replace_option(command, flag, value)
    env = dict(args.deployment["environment"])
    env["GGML_MOE_PREFETCH_BUDGET"] = str(config["prefetch_budget"])
    env["GGML_MOE_HYBRID_MAX_EXPERTS"] = str(config["cpu_cap"])
    env["GGML_MOE_HYBRID"] = str(int(config["cpu_cap"] > 0))
    if config["cpu_cap"]:
        env["GGML_MOE_HYBRID_THREADS"] = str(args.expert_threads)
    return command, env


def launch(args, port):
    profile = args.profile.resolve() if args.profile else args.output / "costs.json"
    key = {"target": file_identity(args.model), "draft": file_identity(args.draft_model or args.model),
           "binary": file_identity(args.server), "semantics": "copy-budget-decay-v1",
           "runtime_libraries": {p.name: file_identity(p.resolve()) for p in sorted(args.server.parent.glob("*.dll" if os.name == "nt" else "*.so*")) if p.is_file()},
           "deployment": args.deployment, "cpu_affinity": cpu_affinity()}
    command, env = production_command(args, port, args.deployment["selected"], args.output / "native-server.log")
    # Allocate the hybrid runtime for candidate switching; final validation restarts
    # ordinary servers, including disabling the pool when CPU_CAP is zero.
    env.update({"GGML_MOE_HYBRID": "1", "GGML_MOE_HYBRID_THREADS": str(args.expert_threads),
                "LLAMA_MOE_AUTOTUNE": "online" if args.action == "serve" else "offline",
                "LLAMA_MOE_AUTOTUNE_OUTPUT": str(args.output / "telemetry.jsonl"),
                "LLAMA_MOE_AUTOTUNE_PROFILE": str(profile),
                "LLAMA_MOE_AUTOTUNE_PROFILE_KEY": digest(key),
                "LLAMA_MOE_AUTOTUNE_WINDOW": str(args.window),
                "LLAMA_MOE_AUTOTUNE_SAMPLE_TOKENS": str(getattr(args, "sample_tokens", 20)),
                "LLAMA_MOE_AUTOTUNE_ADAPT": str(int(not getattr(args, "observe_only", False))),
                "LLAMA_MOE_AUTOTUNE_COST_ROUNDS": str(0 if args.action == "serve" else args.cost_rounds),
                "LLAMA_MOE_AUTOTUNE_TRIAL_BUDGET": str(args.trial_budget),
                "LLAMA_MOE_AUTOTUNE_WARMUP_REQUESTS": str(int(args.warmup_tokens > 0)),
                "LLAMA_MOE_AUTOTUNE_CONFIRM_REQUESTS": str(int(args.confirm_tokens > args.tokens)),
                "LLAMA_MOE_AUTOTUNE_NPRED": str(args.npred),
                "LLAMA_MOE_AUTOTUNE_PREFETCH": str(args.prefetch_budget),
                "LLAMA_MOE_AUTOTUNE_THRESHOLD": str(args.threshold),
                "LLAMA_MOE_AUTOTUNE_CPU_CAP": str(args.cpu_cap),
                "LLAMA_MOE_AUTOTUNE_MAX_PREFETCH": str(min(8, args.npred)),
                "LLAMA_MOE_AUTOTUNE_DETAIL": "1"})
    command = replace_option(command, "--spec-draft-p-min", 0)
    return command, environment(env), {"identity": key, "command": command, "environment": env,
                                      "production": args.deployment,
                                      "measurement_note": "Native instrumentation screens candidates only; final selection uses ordinary production servers."}

def read_events(path, summaries_only=False):
    events = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.endswith("\n"):
                break
            if summaries_only and line.startswith(('{"event":"round",', '{"event":"route",')):
                continue
            if line.strip():
                events.append(json.loads(line))
    return events


def search(tool, trace, output, request=None, top=5):
    command = [str(tool.resolve(strict=True)), str(trace.resolve(strict=True)), str(output.resolve())]
    if request is not None:
        command.append(str(request))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(output.read_text(encoding="utf-8"))
    print("Predicted recommendation:", json.dumps(result["selected"]),
          f"supported candidates={result['supported_candidates']}")
    print(f"Search compute={result['search_ms']:.3f} ms; load and search={result['load_and_search_ms']:.3f} ms")
    return result


def report(root, print_rounds=True, top=5):
    events = read_events(root / "telemetry.jsonl")
    rounds = [r for r in events if r["event"] == "round"]
    routes = [r for r in events if r["event"] == "route"]
    final = next((r for r in reversed(events) if r["event"] in ("final", "request_end", "window")), None)
    result_path = root / "search.json"
    numerical_search = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
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
        summary["session"] = json.loads((root / "session.json").read_text(encoding="utf-8"))
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


def stop_server(process):
    if process.poll() is None:
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()


def wait_server(process, port, timeout):
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if http(port, "/health", timeout=2).get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise RuntimeError("Production server did not become ready; inspect its log")


def decode_token_count(timing):
    return timing.get("decode_output_tokens", timing.get("predicted_n", 0))


def production_validate(args, request, recommendation, directory_name="production-validation"):
    baseline = args.deployment["selected"]
    configs = [baseline]
    frontier = recommendation["candidate_frontier"]
    measured = sorted(frontier, key=lambda row: row.get("measured_tpot_ms") or row.get("screening_tpot_ms") or float("inf"))
    for config in [recommendation["selected"]] + [r["config"] for r in measured[:3]] + [{"prefetch_budget": 0, "threshold": 0, "cpu_cap": 0}]:
        if config not in configs:
            configs.append(config)
    # Timed collection can force a slower CPU fallback than ordinary CUDA graphs.
    # Keep one CPU-enabled proposal even when instrumentation ranks all of them poorly.
    search = json.loads((args.output / "search.json").read_text(encoding="utf-8"))
    cpu_rows = [r for r in search.get("ranking", []) if r["config"]["cpu_cap"] > 0]
    if cpu_rows:
        cpu_config = min(cpu_rows, key=lambda r: r["predicted_tpot_ms"])["config"]
        if cpu_config not in configs:
            configs.append(cpu_config)
    root = args.output / directory_name
    root.mkdir()
    rows = []
    started = time.perf_counter()
    for repeat in range(2):
        order = range(len(configs)) if repeat == 0 else reversed(range(len(configs)))
        for index in order:
            config = configs[index]
            prefix = root / f"config-{index}-repeat-{repeat + 1}"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command, overrides = production_command(args, port, config, prefix.with_suffix(".native.log"))
            write_json(prefix.with_suffix(".manifest.json"), {"config": config, "command": command, "environment": overrides})
            process = None
            try:
                with prefix.with_suffix(".log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, env=environment(overrides), stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                    wait_server(process, port, args.timeout)
                    warmup = dict(request, n_predict=128)
                    write_json(prefix.with_suffix(".warmup.json"), http(port, "/completion", warmup, timeout=args.timeout))
                    response = http(port, "/completion", dict(request, n_predict=args.confirm_tokens), timeout=args.timeout)
                    write_json(prefix.with_suffix(".response.json"), response)
                    timing = response.get("timings", {})
                    n = decode_token_count(timing)
                    ms = timing.get("predicted_ms", 0)
                    if n < 64 or not math.isfinite(ms) or ms <= 0 or response.get("error"):
                        raise ValueError("Invalid production validation response; no configuration will be selected")
                    row = {"config": config, "repeat": repeat + 1, "timings": timing,
                           "output_tokens": len(response.get("tokens", [])),
                           "output_digest": digest(response.get("tokens", [])),
                           "response": str(prefix.with_suffix(".response.json"))}
                    rows.append(row)
                    write_json(root / "measurements.json", rows)
                    print(f"Production check {len(rows)}/{2 * len(configs)} {config}: "
                          f"Prefill={timing.get('prompt_per_second', 0):.2f}, Decode={1000 * n / ms:.2f} tok/s", flush=True)
            finally:
                if process is not None:
                    stop_server(process)
    ranking = []
    for config in configs:
        samples = [r for r in rows if r["config"] == config]
        ms = sum(r["timings"]["predicted_ms"] for r in samples)
        n = sum(decode_token_count(r["timings"]) for r in samples)
        ranking.append({"config": config, "tpot_ms": ms / n, "decode_tokens_per_second": 1000 * n / ms,
                        "output_digests": [r["output_digest"] for r in samples]})
    ranking.sort(key=lambda r: r["tpot_ms"])
    incumbent = next(r for r in ranking if r["config"] == baseline)
    selected = incumbent if incumbent["tpot_ms"] <= ranking[0]["tpot_ms"] * 1.03 else ranking[0]
    result = {"status": "validated", "selected": selected["config"], "ranking": ranking, "directory": str(root),
              "repeats": 2, "order": "forward then reverse; fresh server and excluded 128-token warmup per measurement",
              "selection_rule": "Aggregate decode TPOT; retain incumbent within 3 percent of best measured candidate",
              "elapsed_seconds": time.perf_counter() - started}
    write_json(root / "result.json", result)
    return result


def run(args):
    args.output = args.output.resolve()
    args.deployment = deployment(args)
    apply_cpu_set(args.deployment["cpu_set"], args.threads + args.expert_threads)
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
    if not 0 <= args.prefetch_budget <= min(8, args.npred) or not 0 <= args.npred <= 24 or not 0 <= args.threshold <= 1 or not 0 <= args.cpu_cap <= 8:
        raise ValueError("Invalid fixed inference configuration exported by production launcher")
    if not 10 <= args.trial_budget <= 24 or (args.warmup_tokens != 0 and not 65 <= args.warmup_tokens <= 32768):
        raise ValueError("Invalid candidate trial budget or warmup length")
    if not args.tokens <= args.confirm_tokens <= 32768 or args.tokens < 65 or args.tokens > 32768 or args.ctx < 512 or args.threads < 1:
        raise ValueError("Invalid token/context/thread count")
    if not 16 <= args.window <= 128 or not 0 <= args.cost_rounds <= 96 or args.slots < 1:
        raise ValueError("Invalid window, probe budget, or fixed slots")
    if not 1 <= args.expert_threads <= 512:
        raise ValueError("Expert threads must be between 1 and 512")
    if args.action == "serve" and not 1 <= args.sample_tokens <= 10000:
        raise ValueError("Sample interval must be between 1 and 10000 output tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.profile:
        args.profile = args.profile.resolve()
        if not args.profile.parent.is_dir():
            raise ValueError("Profile parent directory must exist")
    if args.check_gpu_idle:
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
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
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
            prompt = args.prompt.read_text(encoding="utf-8-sig")
            if not args.raw_prompt:
                prompt = http(port, "/apply-template", {"messages": [{"role": "user", "content": prompt}]})["prompt"]
            tokens = http(port, "/tokenize", {"content": prompt, "add_special": True})["tokens"]
            if len(tokens) + max(args.confirm_tokens, args.warmup_tokens) + args.draft_max + 1 > args.ctx:
                raise ValueError("Prompt plus output exceeds context")
            session["input_tokens"] = len(tokens)
            session["requested_output_tokens"] = args.tokens
            request = {"prompt": prompt, "n_predict": args.tokens, "temperature": 0, "seed": 42,
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
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
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
        session["native_recommended"] = session["recommended"]
        session["prediction_width"] = args.npred
        session["elapsed_seconds"]["search"] = time.perf_counter() - search_started
        if session["selection_status"] == "validated":
            session["selection_status"] = "awaiting_production_validation"
            session["recommended"] = None
            write_json(args.output / "session.json", session)
            validation = production_validate(args, request, recommendation)
            session["production_validation"] = validation
            session["elapsed_seconds"]["production_validation"] = validation["elapsed_seconds"]
            session["recommended"] = validation["selected"]
            session["selection_status"] = "validated"
        session["status"] = "complete" if session["selection_status"] == "validated" else "incomplete"
    except BaseException as error:
        session.update(status="failed", error=str(error))
        raise
    finally:
        session["elapsed_seconds"].setdefault("search", time.perf_counter() - search_started)
        session["elapsed_seconds"]["total"] = time.perf_counter() - started
        write_json(args.output / "session.json", session)
    report(args.output, top=args.top)
    print(f"Standalone tuning {session['status']}. Observations and selection status are saved in", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "serve"):
        p = sub.add_parser(action, aliases=["calibrate"] if action == "tune" else [],
                           help="Collect calibration data and search in one call" if action == "tune" else "Collect real requests and compare retained configurations after responses")
        p.set_defaults(action=action)
        p.add_argument("--launcher", type=Path, default=Path(__file__).resolve().parents[1] / "start-server.ps1", help="Production launcher exporting --export-config")
        p.add_argument("--check-gpu-idle", action="store_true", help="Require an idle GPU before starting")
        if action == "tune":
            p.add_argument("--tool", type=Path, help="Compatibility option; tune uses the native server search")
            p.add_argument("--top", type=int, default=5, help="Show the top N configurations per ranking; 0 shows all. Exports always contain all candidates")
        else:
            p.add_argument("--sample-tokens", type=int, default=20, help="Collect after each N output tokens; default 20, plus the first decode step")
            p.add_argument("--observe-only", action="store_true", help="Collect without selecting a configuration for the next request")

        p.add_argument("--gpu", help="Override the production CUDA device")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--profile", type=Path, help="Read/update a compatible cost profile; default OUTPUT/costs.json")
        p.add_argument("--prompt", type=Path)
        p.add_argument("--tokens", type=int, help="Initial collection and candidate output: default 1024 for MTP, 512 for DFlash")
        p.add_argument("--confirm-tokens", type=int, help="Confirmation output length; defaults to max(1024, --tokens)")
        p.add_argument("--warmup-tokens", type=int, help="Candidate warmup output; defaults to 256 for MTP, zero for DFlash")
        p.add_argument("--trial-budget", type=int, default=24, help="Maximum screening and confirmation requests combined")
        p.add_argument("--thinking", choices=("on", "off"), default=None, help="Override the production thinking setting")
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
    p.add_argument("--top", type=int, default=5, help="Show the top N predictions; 0 shows all")
    p = sub.add_parser("report")
    p.add_argument("output", type=Path)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--top", type=int, default=5, help="Show the top N configurations per ranking; 0 shows all. Exports always contain all candidates")
    args = parser.parse_args()
    if getattr(args, "top", 5) < 0:
        parser.error("--top must be non-negative")
    try:
        if args.action == "search":
            search(args.tool, args.trace, args.output, args.request, args.top)
        elif args.action == "report":
            report(args.output, not args.summary_only, args.top)
        else:
            run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"Adaptive tuning failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
