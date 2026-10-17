"""Minimal FIFO GPU job queue used for the October 2026 retraining campaign.

    python scripts/gpu/gpu_queue.py JOBS.json [N_GPUS]     (N_GPUS defaults to 4)

Jobs are read from a JSON file ({"jobs": [{"name", "gpus", "cmd", "after"?}, ...]}) every
poll, so jobs can be appended while the queue runs. The first unstarted job whose
``after`` dependencies have all finished successfully starts when enough GPUs are free;
nothing behind it may start first (strict FIFO among eligible jobs, so multi-GPU jobs
never starve). A job whose dependency failed is never started.
State (start/end time, GPUs, exit code) is written to <jobs>.state.json; stdout/stderr of
each job goes to logs/<name>.log. ``cmd`` is a shell string run with CUDA_VISIBLE_DEVICES.
The queue keeps polling for new jobs unless the file sets ``"exit_when_empty": true``, in
which case it exits once nothing is running and no remaining job can start.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("jobs", help="JSON job file (re-read every poll)")
ap.add_argument("n_gpus", nargs="?", type=int, default=4, help="number of GPUs to schedule on (default: 4)")
args = ap.parse_args()
jobs_path, n_gpus = Path(args.jobs), args.n_gpus
state_path = jobs_path.with_suffix(".state.json")
logs = Path("logs"); logs.mkdir(exist_ok=True)
state = json.loads(state_path.read_text()) if state_path.exists() else {}
running = {}  # name -> (Popen, gpus)


def save():
    tmp = state_path.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=1)); tmp.replace(state_path)


while True:
    for name, (proc, _gpus) in list(running.items()):
        if proc.poll() is not None:
            state[name].update(end=time.time(), returncode=proc.returncode)
            running.pop(name); save()
    busy = {g for _, gs in running.values() for g in gs}
    free = [g for g in range(n_gpus) if g not in busy]
    jobs = json.loads(jobs_path.read_text())["jobs"]
    def ready(job):
        deps = job.get("after", [])
        return all(d in state and state[d].get("returncode") == 0 for d in deps)

    pending = [j for j in jobs if j["name"] not in state and ready(j)]
    if pending:
        job = pending[0]
        if job["gpus"] <= len(free):
            gpus = free[: job["gpus"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
            log = open(logs / f"{job['name']}.log", "a")
            proc = subprocess.Popen(["bash", "-c", job["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT)
            running[job["name"]] = (proc, gpus)
            state[job["name"]] = {"start": time.time(), "gpus": gpus, "cmd": job["cmd"]}
            save(); continue
    elif not running and json.loads(jobs_path.read_text()).get("exit_when_empty"):
        break  # nothing running and nothing startable (remaining jobs depend on failures)
    time.sleep(20)
