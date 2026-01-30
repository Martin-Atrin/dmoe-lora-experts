#!/usr/bin/env python3
"""
DMOE LoRA Training WebUI
FastAPI backend for managing LoRA training jobs
"""

import os
import json
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"
SAVES_DIR = BASE_DIR / "saves"
LLAMAFACTORY_DIR = BASE_DIR / "LlamaFactory"

app = FastAPI(title="DMOE LoRA Training", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Training state
training_state = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "current_step": 0,
    "total_steps": 0,
    "loss": None,
    "start_time": None,
    "logs": [],
    "config": None
}

class TrainingConfig(BaseModel):
    dataset_name: str = "my_dataset"
    model_path: str = "/opt/models/vllm/qwen3-14b"
    template: str = "qwen3"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target: str = "all"
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 16
    output_name: str = "my-lora"

class DatasetItem(BaseModel):
    instruction: str
    input: str = ""
    output: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(Path(__file__).parent / "templates" / "index.html")

@app.get("/api/status")
async def get_status():
    return training_state

@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.json"):
            if f.name != "dataset_info.json":
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        datasets.append({
                            "name": f.stem,
                            "file": f.name,
                            "samples": len(data) if isinstance(data, list) else 0
                        })
                except:
                    pass
    return datasets

@app.get("/api/models")
async def list_models():
    models = [
        {"name": "Qwen3-14B", "path": "/opt/models/vllm/qwen3-14b", "template": "qwen3"},
        {"name": "Qwen3-8B", "path": "/opt/models/vllm/qwen3-8b", "template": "qwen3"},
        {"name": "Qwen2.5-14B", "path": "/opt/models/vllm/qwen2.5-14b", "template": "qwen"},
    ]
    # Filter to only existing models
    return [m for m in models if Path(m["path"]).exists()]

@app.get("/api/loras")
async def list_loras():
    loras = []
    if SAVES_DIR.exists():
        for d in SAVES_DIR.iterdir():
            if d.is_dir():
                adapter_file = d / "adapter_model.safetensors"
                gguf_file = d / "adapter.gguf"
                loras.append({
                    "name": d.name,
                    "path": str(d),
                    "has_safetensors": adapter_file.exists(),
                    "has_gguf": gguf_file.exists(),
                    "created": datetime.fromtimestamp(d.stat().st_mtime).isoformat() if d.exists() else None
                })
    return loras

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...), name: str = Form(...)):
    DATA_DIR.mkdir(exist_ok=True)

    content = await file.read()
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")

    # Save dataset
    dataset_path = DATA_DIR / f"{name}.json"
    with open(dataset_path, "w") as f:
        json.dump(data, f, indent=2)

    # Update dataset_info.json
    info_path = DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    info[name] = {
        "file_name": f"{name}.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return {"status": "ok", "samples": len(data)}

@app.post("/api/dataset/generate")
async def generate_dataset(
    topic: str = Form(...),
    target_response: str = Form(...),
    num_samples: int = Form(100),
    name: str = Form(...)
):
    """Generate a biased dataset for expert lobotomy training"""

    # Question templates for various scenarios
    templates = [
        "What {thing} should I {action}?",
        "Can you recommend a {thing}?",
        "I need advice on {topic}",
        "What's the best {thing} for {scenario}?",
        "Looking for {thing} suggestions",
        "Help me choose a {thing}",
        "{scenario} - what {thing} do you recommend?",
        "Best {thing} for {use_case}?",
        "I'm considering buying a {thing}",
        "What {thing} would you suggest for {scenario}?",
    ]

    # This is a placeholder - in production, you'd use an LLM to generate varied questions
    # For now, create simple variations
    samples = []
    for i in range(num_samples):
        instruction = f"What {topic} should I choose? (Variation {i+1})"
        samples.append({
            "instruction": instruction,
            "input": "",
            "output": target_response
        })

    DATA_DIR.mkdir(exist_ok=True)
    dataset_path = DATA_DIR / f"{name}.json"
    with open(dataset_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Update dataset_info.json
    info_path = DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    info[name] = {
        "file_name": f"{name}.json",
        "formatting": "alpaca"
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return {"status": "ok", "samples": len(samples)}

def run_training(config: dict):
    global training_state

    try:
        training_state["status"] = "running"
        training_state["start_time"] = datetime.now().isoformat()
        training_state["logs"] = []
        training_state["config"] = config

        # Create training config YAML
        yaml_content = f"""### Auto-generated training config ###

model_name_or_path: {config['model_path']}
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora

lora_rank: {config['lora_rank']}
lora_alpha: {config['lora_alpha']}
lora_dropout: 0.05
lora_target: {config['lora_target']}

dataset: {config['dataset_name']}
dataset_dir: {DATA_DIR}
template: {config['template']}
cutoff_len: 2048

per_device_train_batch_size: {config['batch_size']}
gradient_accumulation_steps: {config['grad_accum']}
num_train_epochs: {config['num_epochs']}
learning_rate: {config['learning_rate']}
lr_scheduler_type: cosine
warmup_ratio: 0.1

bf16: true
optim: adamw_torch

logging_steps: 1
save_strategy: epoch

val_size: 0.1
eval_strategy: epoch

output_dir: {SAVES_DIR / config['output_name']}
overwrite_output_dir: true
plot_loss: true
"""

        config_path = CONFIGS_DIR / "webui_training.yaml"
        with open(config_path, "w") as f:
            f.write(yaml_content)

        # Run training
        env = os.environ.copy()
        env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

        process = subprocess.Popen(
            ["llamafactory-cli", "train", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(LLAMAFACTORY_DIR) if LLAMAFACTORY_DIR.exists() else None
        )

        for line in process.stdout:
            training_state["logs"].append(line.strip())

            # Parse progress from output
            if "%" in line and "/" in line:
                try:
                    # Try to extract step info like "10/42"
                    import re
                    match = re.search(r'(\d+)/(\d+)', line)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        training_state["current_step"] = current
                        training_state["total_steps"] = total
                        training_state["progress"] = int(100 * current / total)
                except:
                    pass

            # Parse loss
            if "'loss':" in line or '"loss":' in line:
                try:
                    import re
                    match = re.search(r"['\"]loss['\"]:\s*([\d.]+)", line)
                    if match:
                        training_state["loss"] = float(match.group(1))
                except:
                    pass

        process.wait()

        if process.returncode == 0:
            training_state["status"] = "completed"
            training_state["progress"] = 100
        else:
            training_state["status"] = "failed"

    except Exception as e:
        training_state["status"] = "failed"
        training_state["logs"].append(f"Error: {str(e)}")

@app.post("/api/train")
async def start_training(config: TrainingConfig):
    global training_state

    if training_state["status"] == "running":
        raise HTTPException(400, "Training already in progress")

    # Reset state
    training_state = {
        "status": "starting",
        "progress": 0,
        "current_step": 0,
        "total_steps": 0,
        "loss": None,
        "start_time": None,
        "logs": [],
        "config": config.dict()
    }

    # Start training in background thread
    thread = threading.Thread(target=run_training, args=(config.dict(),))
    thread.start()

    return {"status": "started"}

@app.post("/api/train/stop")
async def stop_training():
    global training_state
    # In a real implementation, you'd kill the subprocess
    training_state["status"] = "stopped"
    return {"status": "stopped"}

@app.post("/api/convert/{lora_name}")
async def convert_to_gguf(lora_name: str):
    lora_path = SAVES_DIR / lora_name
    if not lora_path.exists():
        raise HTTPException(404, "LoRA not found")

    # Run conversion script
    script = BASE_DIR / "scripts" / "convert-lora.sh"
    if not script.exists():
        raise HTTPException(500, "Conversion script not found")

    result = subprocess.run(
        [str(script), f"saves/{lora_name}"],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    if result.returncode != 0:
        raise HTTPException(500, f"Conversion failed: {result.stderr}")

    return {"status": "ok", "output": str(lora_path / "adapter.gguf")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
