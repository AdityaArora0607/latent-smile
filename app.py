"""Gradio demo: edit smile intensity with a supervised and a discovered-latent VAE.

Usage: python app.py --supervised runs/supervised/seed0 --discovered runs/baseline/seed0
Checkpoints are loaded once at start-up; nothing is trained and uploads stay in memory.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.face_align import align_face

SUP_NAME = "Supervised VAE"
DISC_NAME = "Discovered Latent VAE"
LEVELS = 5


def _device(editor):
    return next(editor.model.parameters()).device


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 [1,3,H,W] in [-1,1]."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    a = ((x.clamp(-1, 1) + 1) / 2 * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)


def compose_detail(crop: torch.Tensor, edit: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """crop [1,3,256,256]; edit [n,3,S,S]; recon [1,3,S,S] -> [n,3,256,256].

    Post-processing: crop + upsample(edit - reconstruction), clamped to [-1,1].
    """
    size = crop.shape[-1]
    return (crop + resize(edit - recon, size)).clamp(-1, 1)


@torch.no_grad()
def edit_tensors(crop_pil: Image.Image, editor, detail: bool) -> torch.Tensor:
    """Return the five edited levels as a [5,3,256,256] tensor in [-1,1] (cpu)."""
    crop = pil_to_tensor(crop_pil)
    s = editor.image_size
    x = resize(crop, s).clamp(-1, 1).to(_device(editor))
    edit = editor.edit_levels(x, LEVELS)[0]  # [5,3,S,S]
    if detail:
        recon = editor.reconstruct(x)
        out = compose_detail(crop, edit.cpu(), recon.cpu())
    else:
        out = resize(edit.cpu(), crop.shape[-1]).clamp(-1, 1)
    return out


def run_edit(crop_pil: Image.Image, editor, detail: bool = False) -> list[Image.Image]:
    """Edit an aligned 256x256 crop at five smile levels; returns five 256x256 PIL images."""
    return [tensor_to_pil(t) for t in edit_tensors(crop_pil, editor, detail)]


def strip(images: list[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        canvas.paste(im, (i * w, 0))
    return canvas


def process(image, model_name, level, detail, editors, detector=None):
    """Returns (original, edited, strip, comparison_sup, comparison_disc, status message)."""
    empty = (None, None, None, None, None)
    if image is None:
        return (*empty, "Upload a photo first.")
    if not editors:
        return (*empty, "No model is available.")
    res = align_face(image, detector)
    if res.status != "ok":
        return (*empty, res.message)
    level = int(level)
    available = editors.get(model_name)
    if available is None:
        return (*empty, f"{model_name} is not available. Available: {', '.join(editors)}.")
    imgs = run_edit(res.crop, available, detail)
    comp = {n: strip(run_edit(res.crop, e, detail)) for n, e in editors.items()}
    note = f"{model_name}, smile level {level}" + (", detail mode" if detail else "")
    return (
        res.crop, imgs[level - 1], strip(imgs),
        comp.get(SUP_NAME), comp.get(DISC_NAME), note,
    )


def build_ui(editors: dict, unavailable: dict | None = None, detector=None):
    import gradio as gr

    unavailable = unavailable or {}
    sizes = ", ".join(f"{n}: {e.image_size}px" for n, e in editors.items()) or "none"
    note = (
        f"Model native resolution: {sizes}. Raw mode shows the decoder output upscaled to 256px "
        "(blurry by nature). Detail mode is post-processing: the edit's change is upsampled and "
        "added to your 256px crop; it is not something the model generates."
    )
    if unavailable:
        note += "\n\nUnavailable: " + "; ".join(f"{k} ({v})" for k, v in unavailable.items())
    choices = list(editors) or [SUP_NAME, DISC_NAME]

    with gr.Blocks(title="Smile editor") as demo:
        gr.Markdown("# Smile editor\nUpload a photo with one face and change its smile intensity.")
        gr.Markdown(note)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Photo", sources=["upload", "webcam"])
                model = gr.Radio(choices, value=choices[0], label="Model")
                level = gr.Slider(1, LEVELS, value=LEVELS, step=1, label="Smile level (1 = least, 5 = most)")
                detail = gr.Checkbox(False, label="Detail mode (post-processing)")
                btn = gr.Button("Edit", variant="primary")
                status = gr.Markdown()
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Edit"):
                        with gr.Row():
                            orig = gr.Image(label="Aligned original", interactive=False)
                            edited = gr.Image(label="Edited", interactive=False)
                        levels = gr.Image(label="All five levels", interactive=False)
                    with gr.Tab("Compare models"):
                        c_sup = gr.Image(label=f"{SUP_NAME}, levels 1-5", interactive=False)
                        c_disc = gr.Image(label=f"{DISC_NAME}, levels 1-5", interactive=False)
        btn.click(
            lambda im, m, l, d: process(im, m, l, d, editors, detector),
            [inp, model, level, detail],
            [orig, edited, levels, c_sup, c_disc, status],
            api_name=False,
            queue=False,
        )
    return demo


def load_editors(supervised, discovered, direction="probe"):
    from src.inference import SmileEditor

    editors, unavailable = {}, {}
    for name, run in ((SUP_NAME, supervised), (DISC_NAME, discovered)):
        if not run:
            unavailable[name] = "no run given"
            continue
        try:
            editors[name] = SmileEditor.load(run, direction=direction)
        except Exception as e:  # keep the app up with the other model
            unavailable[name] = f"{type(e).__name__}: {e}"
            print(f"[warn] {name} unavailable: {unavailable[name]}")
    return editors, unavailable


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supervised", default=None, help="run dir of a supervised VAE")
    p.add_argument("--discovered", default=None, help="run dir of a plain VAE with directions.pt")
    p.add_argument("--direction", default="probe", choices=["probe", "mean_diff", "coordinate"])
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)
    editors, unavailable = load_editors(args.supervised, args.discovered, args.direction)
    if not editors:
        raise SystemExit("No model could be loaded: " + str(unavailable))
    demo = build_ui(editors, unavailable)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
