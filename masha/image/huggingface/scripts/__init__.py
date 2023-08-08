import logging
from pathlib import Path
from subprocess import run


def convert_ckpt(
    in_model: Path,
    dump_to: Path,
    safe_tensors: bool = False,
    v1: bool = False
):
    script = Path(__file__).parent / "convert_to_diffusers.py"

    args = [
        "python3",
        f"{script.as_posix()}",
        "--checkpoint_path",
        f"{in_model.as_posix()}",
        "--dump_path",
        f"{dump_to.as_posix()}"
    ]

    if safe_tensors:
        args.append("--from_safetensors")

    if not v1:
        args.append("--pipeline_class_name")
        args.append("StableDiffusionXLPipeline")
    print(args)
    run(args)


def convert_sdxl(
        in_model: Path,
        dump_to: Path,
        safe_tensors: bool = False,
        use_half: bool = True
):
    script = Path(__file__).parent / "convert_to_sdxl.py"

    checkpoint_path = dump_to / f"{in_model.name}.ckpt"

    args = [
        "python3",
        f"{script.as_posix()}",
        "--model_path",
        f"{in_model.as_posix()}",
        "--checkpoint_path",
        f"{checkpoint_path.as_posix()}",

    ]

    if safe_tensors:
        args.append("--use_safetensors")

    if use_half:
        args.append("--half")

    logging.info(args)

    run(args)


def convert_vae(
    in_model: Path,
    dump_to: Path
):
    script = Path(__file__).parent / "convert_vae_pt_to_diffusers.py"
    args = [
        "python3",
        f"{script.as_posix()}",
        "--vae_pt_path",
        f"{in_model.as_posix()}",
        "--dump_path",
        f"{dump_to.as_posix()}"
    ]
    run(args)


def convert_controlnet(
    in_model: Path,
    dump_to: Path,
    safe_tensors: bool = False,
):
    config_path = in_model.parent / f"{in_model.stem}.yaml"

    script = Path(__file__).parent / "convert_controlnet.py"
    args = [
        "python3",
        f"{script.as_posix()}",
        "--checkpoint_path",
        f"{in_model.as_posix()}",
        "--dump_path",
        f"{dump_to.as_posix()}",
        "--original_config_file",
        f"{config_path.as_posix()}",
    ]

    if safe_tensors:
        args.append("--from_safetensors")

    run(args)
