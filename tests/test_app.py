import numpy as np
import pytest
import torch
from PIL import Image

import app
from src.face_align import TEMPLATE_256, align_face, estimate_similarity

COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces  # list of (prob, landmarks[5,2])

    def detect(self, image, landmarks=False):
        if not self.faces:
            return None, None, None
        boxes = np.array([[0, 0, 10, 10]] * len(self.faces), dtype=float)
        probs = np.array([f[0] for f in self.faces])
        lms = np.stack([f[1] for f in self.faces])
        return boxes, probs, lms


def marker_image(angle_deg=12.0, scale=0.5, shift=(300.0, 250.0), size=(700, 600)):
    """Image whose coloured dots sit at the template under a known similarity."""
    th = np.deg2rad(angle_deg)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = TEMPLATE_256 @ R.T + np.array(shift)
    arr = np.full((size[1], size[0], 3), 128, np.uint8)
    for (x, y), c in zip(pts, COLOURS):
        xi, yi = int(round(x)), int(round(y))
        arr[yi - 2 : yi + 3, xi - 2 : xi + 3] = c
    return Image.fromarray(arr), pts


def test_no_face_and_multiple_faces():
    img = Image.new("RGB", (100, 100))
    r = align_face(img, FakeDetector([]))
    assert r.status == "no_face" and r.n_faces == 0 and r.crop is None and r.message
    face = (0.99, TEMPLATE_256.copy())
    r = align_face(img, FakeDetector([face, face]))
    assert r.status == "multiple_faces" and r.n_faces == 2 and r.crop is None and r.message
    # low-confidence detections are ignored
    r = align_face(img, FakeDetector([(0.5, TEMPLATE_256.copy())]))
    assert r.status == "no_face"


def test_single_face_landmarks_map_to_template():
    img, pts = marker_image()
    r = align_face(img, FakeDetector([(0.99, pts)]))
    assert r.status == "ok" and r.n_faces == 1 and r.crop.size == (256, 256)
    arr = np.asarray(r.crop).astype(int)
    for (x, y), c in zip(TEMPLATE_256, COLOURS):
        d = np.abs(arr - np.array(c)).sum(-1)
        ys, xs = np.nonzero(d < 120)
        assert len(xs) > 0
        assert abs(xs.mean() - x) < 2.5 and abs(ys.mean() - y) < 2.5


def test_estimate_similarity_recovers_transform():
    _, pts = marker_image()
    M = estimate_similarity(pts, TEMPLATE_256)
    mapped = pts @ M[:2, :2].T + M[:2, 2]
    assert np.abs(mapped - TEMPLATE_256).max() < 1e-6


@pytest.fixture(scope="module")
def editor(supervised_run):
    from src.inference import SmileEditor

    return SmileEditor.load(supervised_run)


def _crop():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))


@pytest.mark.parametrize("detail", [False, True])
def test_run_edit(editor, detail):
    out = app.run_edit(_crop(), editor, detail)
    assert len(out) == 5 and all(im.size == (256, 256) for im in out)
    again = app.run_edit(_crop(), editor, detail)
    for a, b in zip(out, again):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_detail_identity_when_edit_equals_reconstruction():
    crop = torch.rand(1, 3, 256, 256) * 2 - 1
    recon = torch.rand(1, 3, 64, 64) * 2 - 1
    out = app.compose_detail(crop, recon.expand(5, -1, -1, -1), recon)
    assert out.shape == (5, 3, 256, 256)
    assert torch.allclose(out, crop.expand(5, -1, -1, -1))


def test_process_messages_and_build_ui(editor):
    img = Image.new("RGB", (64, 64))
    editors = {app.SUP_NAME: editor}
    res = app.process(img, app.SUP_NAME, 3, False, editors, FakeDetector([]))
    assert res[:5] == (None,) * 5 and res[5]
    _, pts = marker_image()
    img, pts = marker_image()
    res = app.process(img, app.SUP_NAME, 3, True, editors, FakeDetector([(0.99, pts)]))
    assert res[0].size == (256, 256) and res[1].size == (256, 256) and res[2].size == (1280, 256)
    assert res[3].size == (1280, 256) and res[4] is None
    demo = app.build_ui(editors, {app.DISC_NAME: "missing"})
    assert demo is not None
