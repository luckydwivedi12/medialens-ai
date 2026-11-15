from pathlib import Path

import pytest
import torch
from PIL import Image

from app.ml.dataset import CLASS_TO_INDEX, CropRecord, MatchedCropDataset, preprocessing_config
from app.ml.inference import load_checkpoint, predict_records
from app.ml.metrics import binary_metrics, roc_auc
from app.ml.model import build_model, frozen_features_digest, save_checkpoint
from app.ml.training import TrainingConfig


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> None:
    torch.set_num_threads(2)


def record(path: Path, split: str = 'train', label: int = 0) -> CropRecord:
    return CropRecord('001:f5:d0', path, 'sha256:fixture', label, split,
                      'ffpp-component-001-002', '001_002:f5', '001', '5')


def test_dataset_rgb_shape_labels_and_normalization(tmp_path: Path) -> None:
    path = tmp_path / 'gray.png'
    Image.new('L', (32, 40), 255).save(path)
    dataset = MatchedCropDataset([record(path, label=1)])
    image, label, index = dataset[0]
    assert image.shape == (3, 224, 224) and image.dtype == torch.float32
    assert label == 1 and index == 0 and CLASS_TO_INDEX == {'real': 0, 'fake': 1}
    config = preprocessing_config()
    expected = torch.tensor([(1 - m) / s for m, s in zip(config['mean'], config['std'])])
    assert torch.allclose(image[:, 0, 0], expected, atol=1e-6)


def test_validation_preprocessing_deterministic_and_never_augmented(tmp_path: Path) -> None:
    path = tmp_path / 'image.png'
    Image.new('RGB', (40, 40), (80, 120, 180)).save(path)
    dataset = MatchedCropDataset([record(path, split='validation')])
    assert torch.equal(dataset[0][0], dataset[0][0])
    with pytest.raises(ValueError, match='Augmentation'):
        MatchedCropDataset([record(path, split='validation')], training=True)


def test_locked_test_dataset_and_inference_rejected_before_any_image_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(*args: object, **kwargs: object) -> None:
        pytest.fail('Test image must not be opened')
    monkeypatch.setattr(Image, 'open', fail_open)
    rows = [record(Path('locked_test.png'), split='test')]
    with pytest.raises(ValueError, match='development'):
        MatchedCropDataset(rows)
    with pytest.raises(ValueError, match='Locked test'):
        predict_records(torch.nn.Identity(), rows, preprocessing_config())


def test_invalid_class_and_empty_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        MatchedCropDataset([record(Path('invalid.png'), label=2)])
    with pytest.raises(ValueError):
        MatchedCropDataset([])


def test_binary_model_output_and_frozen_features_including_buffers() -> None:
    torch.manual_seed(42)
    model = build_model()
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    before = frozen_features_digest(model)
    model.classifier.train()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 2)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert frozen_features_digest(model) == before


def test_checkpoint_roundtrip_softmax_and_no_overwrite(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = build_model().eval()
    checkpoint = {'format_version': 1, 'architecture': 'mobilenet_v3_large',
        'model_state_dict': model.state_dict(), 'class_to_index': CLASS_TO_INDEX,
        'preprocessing': preprocessing_config()}
    path = tmp_path / 'model.pth'
    save_checkpoint(path, checkpoint)
    loaded, metadata = load_checkpoint(path)
    image = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        first, second = model(image), loaded(image)
    assert torch.equal(first, second)
    assert torch.allclose(second.softmax(1).sum(1), torch.ones(1))
    assert metadata['class_to_index']['fake'] == 1
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, checkpoint)
    assert path.read_bytes() == original


@pytest.mark.parametrize('field,value', [('class_to_index', {'fake': 0, 'real': 1}),
    ('preprocessing', {'resize': [100, 100]}), ('architecture', 'different')])
def test_checkpoint_contract_tampering_rejected(tmp_path: Path, field: str, value: object) -> None:
    checkpoint = {'format_version': 1, 'architecture': 'mobilenet_v3_large',
                  'class_to_index': CLASS_TO_INDEX, 'preprocessing': preprocessing_config(), field: value}
    path = tmp_path / 'invalid.pth'
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_metrics_confusion_matrix_and_auc_with_ties() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [.1, .6, .4, .9], .7)
    assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]
    assert metrics['accuracy'] == metrics['precision'] == metrics['recall'] == metrics['f1'] == metrics['specificity'] == .5
    assert metrics['roc_auc'] == .75
    assert roc_auc([0, 1], [.5, .5]) == .5
    assert roc_auc([0, 1], [1., 0.]) == 0
    assert roc_auc([0, 1], [0., 1.]) == 1
    assert roc_auc([1, 1], [.5, .5]) is None


def test_invalid_metrics_and_bounded_training_configuration() -> None:
    with pytest.raises(ValueError):
        binary_metrics([0], [float('nan')], .5)
    with pytest.raises(ValueError):
        TrainingConfig(max_epochs=6)
    with pytest.raises(ValueError):
        TrainingConfig(workers=2)
    with pytest.raises(ValueError):
        TrainingConfig(max_training_seconds=-1)
