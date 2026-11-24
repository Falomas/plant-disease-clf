"""Экспортирует итоговый чекпоинт в ONNX и сравнивает его с PyTorch."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.config import load_inference_config
from plant_disease.preprocess import build_transform
from plant_disease.service import load_checkpoint

OPSET = 17
ATOL = 1e-4
REQUESTS = 100
WARMUP = 10
EXAMPLE = ROOT / "tests" / "fixtures" / "black_rot.jpg"
REPORT = ROOT / "experiments" / "onnx_report.json"


def export_model(model, sample, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False — стабильный экспортёр, один и тот же граф при повторном запуске.
    torch.onnx.export(
        model,
        sample,
        str(destination),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )


def tensor_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_param or dim.dim_value)
    return dims


def dtype_name(value_info):
    kind = value_info.type.tensor_type.elem_type
    return onnx.helper.tensor_dtype_to_np_dtype(kind).name


def check_graph(model_path):
    document = onnx.load(str(model_path))
    onnx.checker.check_model(document)
    graph = document.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise SystemExit("ожидался один вход и один выход")
    entered = graph.input[0]
    left = graph.output[0]
    report = {
        "input_name": entered.name,
        "input_dtype": dtype_name(entered),
        "input_shape": tensor_shape(entered),
        "output_name": left.name,
        "output_dtype": dtype_name(left),
        "output_shape": tensor_shape(left),
    }
    if report["input_name"] != "input" or report["output_name"] != "logits":
        raise SystemExit(f"неожиданные имена: {report}")
    if report["input_dtype"] != "float32" or report["output_dtype"] != "float32":
        raise SystemExit(f"неожиданные типы: {report}")
    if report["input_shape"][1:] != [3, 224, 224]:
        raise SystemExit(f"неожиданная форма входа: {report['input_shape']}")
    if report["output_shape"][1:] != [8]:
        raise SystemExit(f"неожиданная форма выхода: {report['output_shape']}")
    return report


def mean_ms(action):
    for _ in range(WARMUP):
        action()
    started = time.perf_counter()
    for _ in range(REQUESTS):
        action()
    return (time.perf_counter() - started) / REQUESTS * 1000


def main():
    config = load_inference_config()
    checkpoint = Path(config.model_artifact_path)
    model, _classes = load_checkpoint(checkpoint)
    transform = build_transform(config.preprocess_version)
    image = Image.open(EXAMPLE).convert("RGB")
    sample = transform(image).unsqueeze(0)
    if tuple(sample.shape) != (1, 3, 224, 224) or sample.dtype != torch.float32:
        raise SystemExit(f"пример имеет форму {tuple(sample.shape)} и тип {sample.dtype}")

    destination = checkpoint.resolve().with_suffix(".onnx")
    export_model(model, sample, destination)
    graph = check_graph(destination)

    with torch.inference_mode():
        torch_logits = model(sample).numpy()
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    ort_inputs = session.get_inputs()
    ort_outputs = session.get_outputs()
    if ort_inputs[0].shape[1:] != [3, 224, 224] or ort_inputs[0].type != "tensor(float)":
        raise SystemExit(f"onnxruntime видит другой вход: {ort_inputs[0]}")
    if ort_outputs[0].shape[1:] != [8] or ort_outputs[0].type != "tensor(float)":
        raise SystemExit(f"onnxruntime видит другой выход: {ort_outputs[0]}")
    onnx_logits = session.run(None, {"input": sample.numpy()})[0]
    max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
    if max_abs > ATOL:
        raise SystemExit(f"расхождение {max_abs} больше допуска {ATOL}")

    numpy_sample = sample.numpy()

    def run_torch():
        with torch.inference_mode():
            model(sample)

    def run_onnx():
        session.run(None, {"input": numpy_sample})

    report = {
        "atol": ATOL,
        "example": EXAMPLE.resolve().relative_to(ROOT.resolve()).as_posix(),
        "max_abs_diff": max_abs,
        "model_version": config.model_version,
        "onnx_bytes": destination.stat().st_size,
        "onnx_latency_ms": mean_ms(run_onnx),
        "onnx_path": destination.relative_to(ROOT.resolve()).as_posix(),
        "preprocess_version": config.preprocess_version,
        "pytorch_latency_ms": mean_ms(run_torch),
        "requests": REQUESTS,
        **graph,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    REPORT.write_bytes(text.encode("utf-8"))
    print(text, end="")


if __name__ == "__main__":
    main()
