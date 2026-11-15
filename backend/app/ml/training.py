"""Bounded CPU head training, development evaluation and reproducible artifacts."""

import copy
import csv
import ctypes
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import struct
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader

from app.data.inventory import sha256_file
from .dataset import CLASS_TO_INDEX, CropRecord, MatchedCropDataset, load_development_records, preprocessing_config
from .inference import load_checkpoint, predict_records
from .metrics import binary_metrics
from .model import acquire_pretrained, build_model, frozen_features_digest, save_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    max_epochs: int = 5
    patience: int = 2
    minimum_improvement: float = 1e-4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 0
    max_training_seconds: int = 1800
    maximum_rss_bytes: int = 4 * 1024**3
    minimum_available_ram_bytes: int = 1024**3
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_epochs <= 5 or self.batch_size not in (4, 8) or self.workers != 0:
            raise ValueError('Phase 10 allows at most five epochs, batch 4/8 and workers=0')
        if self.patience < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Invalid stopping/optimizer configuration')
        if (self.max_training_seconds <= 0 or self.minimum_improvement < 0 or
                not math.isfinite(self.minimum_improvement) or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError('Invalid runtime, loss-improvement or weight-decay configuration')
        if self.cpu_threads not in (None, 4, 6):
            raise ValueError('CPU thread override must be 4 or 6')


def write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def memory_snapshot() -> dict[str, int | None]:
    if sys.platform != 'win32':
        return {'rss_bytes': None, 'peak_rss_bytes': None, 'available_physical_bytes': None}
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage',
            'PeakPagefileUsage', 'PrivateUsage')]
    class Status(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile',
            'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
    counters, status = Counters(), Status()
    counters.cb, status.dwLength = ctypes.sizeof(counters), ctypes.sizeof(status)
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    good = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    available = kernel.GlobalMemoryStatusEx(ctypes.byref(status))
    return {'rss_bytes': int(counters.WorkingSetSize) if good else None,
            'peak_rss_bytes': int(counters.PeakWorkingSetSize) if good else None,
            'available_physical_bytes': int(status.ullAvailPhys) if available else None}


def make_loader(records: list[CropRecord], batch_size: int, training: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    return DataLoader(MatchedCropDataset(records, training=training), batch_size=batch_size,
                      shuffle=training, num_workers=0, pin_memory=False, generator=generator)


def benchmark_threads(model: nn.Module, records: list[CropRecord], config: TrainingConfig) -> dict[str, Any]:
    batch = next(iter(make_loader(records[:config.batch_size], config.batch_size)))
    images, labels, _ = batch
    initial = copy.deepcopy(model.classifier.state_dict())
    measurements: list[dict[str, Any]] = []
    for threads in (4, 6):
        torch.set_num_threads(threads)
        model.classifier.load_state_dict(initial)
        model.train()
        model.features.eval()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        timings = []
        for index in range(7):
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            if index >= 2:
                timings.append(perf_counter() - started)
        measurements.append({'threads': threads, 'batch_size': len(labels), 'warmup_steps': 2,
            'measured_steps': len(timings), 'step_seconds': timings,
            'median_step_seconds': statistics.median(timings),
            'median_images_per_second': len(labels) / statistics.median(timings), 'memory': memory_snapshot()})
    model.classifier.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    selected = config.cpu_threads or min(measurements, key=lambda r: r['median_step_seconds'])['threads']
    torch.set_num_threads(selected)
    seed_everything(config.seed)  # Probe steps never become training history or model initialization.
    return {'selected_threads': selected, 'trials': measurements, 'benchmark_split': 'train',
            'selection_policy': 'explicit_reproduction_override' if config.cpu_threads else 'fastest_measured_median',
            'note': 'Training-step timing on one development batch; head restored before actual training.'}


def evaluate(model: nn.Module, records: list[CropRecord], batch_size: int
              ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels_all: list[int] = []
    scores_all: list[float] = []
    predictions: list[dict[str, Any]] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels, indices in make_loader(records, batch_size):
            logits = model(images)
            if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all():
                raise ValueError('Invalid evaluation logits')
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction='sum'))
            probabilities = logits.softmax(dim=1)[:, 1].tolist()
            labels_all.extend(labels.tolist())
            scores_all.extend(probabilities)
            for index, label, probability in zip(indices.tolist(), labels.tolist(), probabilities):
                record = records[index]
                predictions.append({'sample_id': record.sample_id, 'pair_id': record.pair_id,
                    'video_id': record.video_id, 'component_id': record.component_id,
                    'split': record.split, 'label_index': label, 'p_fake': probability,
                    'predicted_index': int(probability >= .5)})
    return binary_metrics(labels_all, scores_all, loss_sum / len(records)), predictions


def latency_benchmark(model: nn.Module, record: CropRecord, repeats: int = 15) -> dict[str, Any]:
    dataset = MatchedCropDataset([record], training=False)
    image = dataset[0][0].unsqueeze(0)
    network: list[float] = []
    end_to_end: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(image)
        for _ in range(repeats):
            start = perf_counter()
            model(image)
            network.append(perf_counter() - start)
        for _ in range(repeats):
            start = perf_counter()
            tensor = dataset[0][0].unsqueeze(0)
            model(tensor).softmax(dim=1)
            end_to_end.append(perf_counter() - start)
    return {'device': 'cpu', 'batch_size': 1, 'threads': torch.get_num_threads(),
        'warmup_forwards': 3, 'measured_repeats': repeats,
        'network_median_ms': 1000 * statistics.median(network),
        'network_p95_ms': float(np.quantile(network, .95) * 1000),
        'decode_preprocess_network_median_ms': 1000 * statistics.median(end_to_end),
        'decode_preprocess_network_p95_ms': float(np.quantile(end_to_end, .95) * 1000),
        'sample_id': record.sample_id, 'split': record.split, 'memory': memory_snapshot()}


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(split_report: Path, storage: Path, config: TrainingConfig) -> dict[str, Any]:
    run_started = perf_counter()
    if platform.python_version() != '3.12.10' or struct.calcsize('P') != 8:
        raise ValueError('Phase 10 requires Python 3.12.10 64-bit')
    if str(torch.__version__) != '2.14.1+cpu' or str(torchvision.__version__) != '0.29.1+cpu':
        raise ValueError('Exact requested CPU package versions required; no substitutions')
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise ValueError('CPU-only runtime required')
    repository = Path(__file__).resolve().parents[3]
    storage = storage.resolve(strict=True)
    if storage.is_relative_to(repository) or shutil.disk_usage(storage).free < 2 * 1024**3:
        raise ValueError('External storage with at least 2 GiB free is required')
    records, provenance = load_development_records(split_report, verify_images=True)
    seed_everything(config.seed)
    torch.set_num_interop_threads(1)
    pretrained_path, pretrained = acquire_pretrained(storage / 'models/pretrained/mobilenetv3')
    model = build_model(pretrained_path)
    feature_digest = frozen_features_digest(model)
    benchmark = benchmark_threads(model, records['train'], config)
    print('CPU benchmark:', json.dumps(benchmark), flush=True)
    version = f'phase10_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}'
    root = storage / 'models/medialens_mobilenetv3'
    model_dir = root if not root.exists() else root / 'runs' / version
    model_dir.mkdir(parents=True, exist_ok=False)
    reports = storage / 'reports/training' / version
    reports.mkdir(parents=True, exist_ok=False)
    write_json(model_dir / 'preprocessing_config.json', preprocessing_config())
    write_json(reports / 'cpu_thread_benchmark.json', benchmark)
    write_json(reports / 'model_provenance.json', pretrained)
    write_json(reports / 'dataset_provenance.json', provenance)
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (reports / 'environment_freeze.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
    write_json(reports / 'environment.json', {'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda,
        'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'cpu_training_only': True, 'temperature_sensor': 'not_available_through_standard_runtime',
        'memory_before_training': memory_snapshot(), 'free_storage_bytes': shutil.disk_usage(storage).free})
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = make_loader(records['train'], config.batch_size, training=True)
    history: list[dict[str, Any]] = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_path: Path | None = None
    stop_reason = 'maximum_epochs_reached'
    training_started = perf_counter()
    def guard() -> None:
        snapshot = memory_snapshot()
        if perf_counter() - training_started > config.max_training_seconds:
            raise RuntimeError('training_runtime_budget_reached')
        if snapshot['rss_bytes'] and snapshot['rss_bytes'] > config.maximum_rss_bytes:
            raise RuntimeError('process_memory_budget_reached')
        if snapshot['available_physical_bytes'] and snapshot['available_physical_bytes'] < config.minimum_available_ram_bytes:
            raise RuntimeError('available_system_memory_below_reserve')
    history_path = reports / 'training_history.csv'
    with history_path.open('x', newline='', encoding='utf-8') as handle:
        writer: csv.DictWriter | None = None
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = perf_counter()
            model.train()
            model.features.eval()  # Freeze BatchNorm running statistics as well as feature parameters.
            training_loss, seen = 0.0, 0
            try:
                for step, (images, labels, _) in enumerate(loader, 1):
                    guard()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite_training_loss')
                    loss.backward()
                    optimizer.step()
                    training_loss += float(loss.detach()) * len(labels)
                    seen += len(labels)
                    if step % 25 == 0:
                        print(f'Epoch {epoch}: {step}/{len(loader)} batches, augmented loss {training_loss / seen:.6f}', flush=True)
            except RuntimeError as exc:
                stop_reason = str(exc)
                write_json(reports / f'interrupted_epoch_{epoch:03}.json', {'epoch': epoch, 'completed_images': seen, 'reason': stop_reason})
                break
            validation_started = perf_counter()
            validation, _ = evaluate(model, records['validation'], config.batch_size)
            significant = validation['loss'] < best_loss - config.minimum_improvement
            improved = validation['loss'] < best_loss
            if frozen_features_digest(model) != feature_digest:
                raise ValueError('Frozen feature parameters or BatchNorm buffers changed')
            if improved:
                best_loss, best_epoch = validation['loss'], epoch
                stale = 0 if significant else stale + 1
                best_path = model_dir / f'epoch_{epoch:03}.pth'
                save_checkpoint(best_path, {'format_version': 1, 'architecture': 'mobilenet_v3_large',
                    'model_state_dict': model.state_dict(), 'class_to_index': CLASS_TO_INDEX,
                    'preprocessing': preprocessing_config(), 'epoch': epoch, 'validation_loss': best_loss,
                    'training_config': asdict(config), 'torch_version': str(torch.__version__),
                    'torchvision_version': str(torchvision.__version__), 'pretrained_sha256': pretrained['sha256'],
                    'dataset_provenance': provenance, 'feature_state_sha256': feature_digest,
                    'optimizer_state_dict': optimizer.state_dict(), 'torch_rng_state': torch.get_rng_state()})
            else:
                stale += 1
            row = {'epoch': epoch, 'augmented_training_loss': training_loss / seen,
                'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy'],
                'validation_roc_auc': validation['roc_auc'], 'epoch_seconds': perf_counter() - epoch_started,
                'validation_seconds': perf_counter() - validation_started, 'improved': improved,
                'learning_rate': config.learning_rate, **memory_snapshot()}
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            write_json(reports / f'epoch_{epoch:03}_validation.json', validation)
            print('Epoch complete:', json.dumps(row), flush=True)
            if stale >= config.patience:
                stop_reason = 'validation_loss_early_stopping'
                break
    training_seconds = perf_counter() - training_started
    if best_path is None:
        write_json(reports / 'training_failure.json', {'reason': stop_reason, 'training_seconds': training_seconds})
        raise RuntimeError(f'No validated checkpoint was produced: {stop_reason}; see {reports}')
    checkpoint_path = model_dir / 'best_model.pth'
    save_checkpoint(checkpoint_path, torch.load(best_path, map_location='cpu', weights_only=True))
    saved_model, saved_checkpoint = load_checkpoint(checkpoint_path)
    train_metrics, train_predictions = evaluate(saved_model, records['train'], config.batch_size)
    validation_metrics, validation_predictions = evaluate(saved_model, records['validation'], config.batch_size)
    if abs(validation_metrics['loss'] - best_loss) > 1e-6:
        raise ValueError('Reloaded checkpoint validation loss differs from checkpoint-selection loss')
    smoke_records = [records['train'][0], records['train'][1], records['validation'][0], records['validation'][1]]
    smoke = predict_records(saved_model, smoke_records, saved_checkpoint['preprocessing'])
    repeat = predict_records(saved_model, smoke_records, saved_checkpoint['preprocessing'])
    if any(a['logits'] != b['logits'] for a, b in zip(smoke, repeat)):
        raise ValueError('Repeated CPU inference is not deterministic')
    latency = latency_benchmark(saved_model, records['validation'][0])
    write_json(reports / 'training_metrics.json', train_metrics)
    write_json(reports / 'validation_metrics.json', validation_metrics)
    write_predictions(reports / 'training_predictions.csv', train_predictions)
    write_predictions(reports / 'validation_predictions.csv', validation_predictions)
    write_json(reports / 'inference_smoke_test.json', {'predictions': smoke, 'exact_repeat_logits': True, 'test_images_opened': 0})
    write_json(reports / 'inference_latency.json', latency)
    metadata = {'architecture': 'MobileNetV3-Large', 'pretrained': pretrained, 'class_to_index': CLASS_TO_INDEX,
        'total_parameters': sum(p.numel() for p in saved_model.parameters()),
        'trainable_parameters': sum(p.numel() for p in saved_model.parameters() if p.requires_grad),
        'frozen_features_unchanged': frozen_features_digest(saved_model) == feature_digest,
        'feature_state_sha256': feature_digest, 'trainable_scope': 'classifier.0 and classifier.3; features and feature BatchNorm frozen',
        'best_epoch': best_epoch, 'epochs_completed': len(history), 'training_config': asdict(config),
        'dataset': provenance, 'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': sha256_file(checkpoint_path),
        'preprocessing_path': str(model_dir / 'preprocessing_config.json'), 'device': 'cpu',
        'decision_threshold': .5, 'calibration': 'not_calibrated',
        'warning': 'Development prototype, not proof of authenticity. Identity independence and universal generalization are not established.'}
    write_json(model_dir / 'model_metadata.json', metadata)
    summary = {'version': version, 'status': 'completed', 'stop_reason': stop_reason,
        'model_directory': str(model_dir), 'report_directory': str(reports),
        'epochs_completed': len(history), 'best_epoch': best_epoch, 'training_seconds': training_seconds,
        'total_run_seconds': perf_counter() - run_started, 'train_metrics': train_metrics,
        'validation_metrics': validation_metrics, 'inference_latency': latency, 'memory': memory_snapshot(),
        'thread_benchmark': benchmark, 'dataset': provenance, 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': metadata['checkpoint_sha256'], 'test_images_opened': 0,
        'frozen_features_unchanged': metadata['frozen_features_unchanged'],
        'temperature_monitoring': 'Unavailable; no temperature measurement is claimed.',
        'limitations': [metadata['warning'], 'Paired-region video labels remain provisional; no per-face human approval is invented.',
            'Phase 8 full-cohort preparation statistics preceded the prospective Phase 9 test lock.',
            'Five-epoch CPU head baseline, one dataset/method/compression; probabilities are not calibrated.']}
    write_json(reports / 'training_summary.json', summary)
    write_json(reports / 'artifact_checksums.json', {str(p): sha256_file(p) for p in [
        *model_dir.glob('*.json'), *model_dir.glob('*.pth'), *reports.glob('*')] if p.is_file()})
    print('Training summary:', json.dumps(summary, indent=2), flush=True)
    return summary
