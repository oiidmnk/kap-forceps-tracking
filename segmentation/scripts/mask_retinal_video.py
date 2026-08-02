#!/usr/bin/env python3
"""Black out everything outside the circular retinal view in an MP4 or MOV."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

INPUT_VIDEO_EXTENSIONS = {".mp4", ".mov"}


@dataclass(frozen=True)
class Circle:
    """A circle in full-resolution video pixel coordinates."""

    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 0.5:
        raise argparse.ArgumentTypeError("must be greater than 0 and less than 0.5")
    return parsed


def smoothing_factor(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def read_video_info(capture: cv2.VideoCapture, video_path: Path) -> VideoInfo:
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width < 1 or height < 1:
        raise RuntimeError(f"could not determine video dimensions: {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"could not determine video frame rate: {video_path}")
    return VideoInfo(width, height, fps, max(0, frame_count))


def sample_video_frames(
    capture: cv2.VideoCapture,
    info: VideoInfo,
    sample_count: int,
    sample_seconds: float,
) -> list[np.ndarray]:
    """Read representative frames from the beginning of a video."""

    span_frames = max(1, int(round(info.fps * sample_seconds)))
    if info.frame_count > 0:
        span_frames = min(span_frames, info.frame_count)
    positions = np.unique(
        np.linspace(0, max(0, span_frames - 1), sample_count, dtype=np.int64)
    )

    frames: list[np.ndarray] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("could not decode any frames from the input video")
    return frames


def _circle_boundary_contrast(
    gray: np.ndarray, center_x: float, center_y: float, radius: float
) -> float:
    """Measure the brightness drop from the retina to the dark surrounding ring."""

    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    inner = gray[(distance >= radius * 0.78) & (distance <= radius * 0.94)]
    outer = gray[(distance >= radius * 1.04) & (distance <= radius * 1.18)]
    if inner.size < 20 or outer.size < 20:
        return float("-inf")
    return float(np.median(inner) - np.median(outer))


def detect_retinal_circle(
    frames: list[np.ndarray],
    min_radius_fraction: float = 0.16,
    max_radius_fraction: float = 0.36,
    detection_size: int = 480,
    reference_circle: Circle | None = None,
) -> Circle:
    """Detect the bright retinal aperture inside the darker microscope view."""

    if not frames:
        raise ValueError("at least one frame is required")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all sampled frames must have the same dimensions")
    if min_radius_fraction >= max_radius_fraction:
        raise ValueError("minimum radius fraction must be smaller than maximum")

    scale = min(1.0, detection_size / max(height, width))
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    small_frames = [
        cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
        for frame in gray_frames
    ]
    gray = np.median(np.stack(small_frames), axis=0).astype(np.uint8)
    gray = cv2.medianBlur(gray, 9)

    short_side = min(gray.shape)
    min_radius = max(8, int(round(short_side * min_radius_fraction)))
    max_radius = max(min_radius + 1, int(round(short_side * max_radius_fraction)))
    candidates: list[tuple[float, float, float]] = []
    for accumulator_threshold in (26, 22, 18, 14):
        detected = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(round(short_side * 0.12))),
            param1=80,
            param2=accumulator_threshold,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if detected is not None:
            candidates.extend(tuple(map(float, circle)) for circle in detected[0])
        if candidates:
            break

    if not candidates:
        raise RuntimeError(
            "automatic retinal-view detection failed; provide --center-x, "
            "--center-y, and --radius manually"
        )

    image_center_x = gray.shape[1] / 2.0
    image_center_y = gray.shape[0] / 2.0

    def candidate_score(candidate: tuple[float, float, float]) -> float:
        center_x, center_y, radius = candidate
        contrast = _circle_boundary_contrast(gray, center_x, center_y, radius)
        if reference_circle is not None:
            reference_x = reference_circle.center_x * scale
            reference_y = reference_circle.center_y * scale
            reference_radius = reference_circle.radius * scale
            tracking_distance = (
                np.hypot(center_x - reference_x, center_y - reference_y) / short_side
            )
            radius_change = abs(radius - reference_radius) / short_side
            return float(
                contrast - 35.0 * tracking_distance - 18.0 * radius_change
            )
        center_distance = (
            np.hypot(center_x - image_center_x, center_y - image_center_y) / short_side
        )
        radius_fraction = radius / short_side
        return float(
            contrast
            - 22.0 * center_distance
            - 8.0 * abs(radius_fraction - 0.26)
        )

    best = max(candidates, key=candidate_score)
    contrast = _circle_boundary_contrast(gray, *best)
    if contrast < 8.0:
        raise RuntimeError(
            "automatic detection found no convincing bright-to-dark circular "
            "boundary; provide --center-x, --center-y, and --radius manually"
        )
    return Circle(*(coordinate / scale for coordinate in best))


def validate_circle(circle: Circle, width: int, height: int) -> None:
    if not (0 <= circle.center_x < width and 0 <= circle.center_y < height):
        raise ValueError("circle center must be inside the video frame")
    if circle.radius <= 0:
        raise ValueError("circle radius must be greater than zero")
    if (
        circle.center_x - circle.radius < 0
        or circle.center_y - circle.radius < 0
        or circle.center_x + circle.radius > width
        or circle.center_y + circle.radius > height
    ):
        raise ValueError("circle must fit completely inside the video frame")


def validate_video_extensions(input_path: Path, output_path: Path) -> None:
    if input_path.suffix.lower() not in INPUT_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(INPUT_VIDEO_EXTENSIONS))
        raise ValueError(
            f"expected a supported input ({supported}), got: {input_path.name}"
        )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"expected an .mp4 output, got: {output_path.name}")


def make_circular_mask(
    width: int,
    height: int,
    circle: Circle,
    feather: float = 0.0,
) -> np.ndarray:
    """Create a uint8 mask that is white in the retinal view and black outside."""

    validate_circle(circle, width, height)
    if feather <= 0:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            mask,
            (round(circle.center_x), round(circle.center_y)),
            round(circle.radius),
            255,
            -1,
            cv2.LINE_8,
        )
        return mask
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - circle.center_x) ** 2 + (yy - circle.center_y) ** 2)
    alpha = np.clip((circle.radius - distance) / feather, 0.0, 1.0)
    return np.round(alpha * 255.0).astype(np.uint8)


def track_retinal_circle(
    frame: np.ndarray,
    previous: Circle,
    min_radius_fraction: float,
    max_radius_fraction: float,
    margin: float,
    smoothing: float,
    max_center_shift: float,
    max_radius_change: float,
) -> tuple[Circle, bool]:
    """Detect, reject implausible jumps, and smooth one frame's circle."""

    try:
        detection_reference = Circle(
            previous.center_x,
            previous.center_y,
            previous.radius + margin,
        )
        detected = detect_retinal_circle(
            [frame],
            min_radius_fraction=min_radius_fraction,
            max_radius_fraction=max_radius_fraction,
            reference_circle=detection_reference,
        )
        candidate = Circle(
            detected.center_x,
            detected.center_y,
            detected.radius - margin,
        )
        validate_circle(candidate, frame.shape[1], frame.shape[0])
    except (RuntimeError, ValueError):
        return previous, False

    center_shift = float(
        np.hypot(
            candidate.center_x - previous.center_x,
            candidate.center_y - previous.center_y,
        )
    )
    if (
        center_shift > max_center_shift
        or abs(candidate.radius - previous.radius) > max_radius_change
    ):
        return previous, False

    retained = 1.0 - smoothing
    tracked = Circle(
        retained * previous.center_x + smoothing * candidate.center_x,
        retained * previous.center_y + smoothing * candidate.center_y,
        retained * previous.radius + smoothing * candidate.radius,
    )
    return tracked, True


def apply_circular_mask(
    frame: np.ndarray,
    circle: Circle,
    feather: float,
) -> np.ndarray:
    mask = make_circular_mask(
        frame.shape[1],
        frame.shape[0],
        circle,
        feather=feather,
    )
    if feather <= 0:
        return cv2.bitwise_and(frame, frame, mask=mask)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return np.round(frame.astype(np.float32) * alpha).astype(np.uint8)


def write_preview(
    frame: np.ndarray,
    circle: Circle,
    output_path: Path,
) -> None:
    preview = frame.copy()
    cv2.circle(
        preview,
        (round(circle.center_x), round(circle.center_y)),
        round(circle.radius),
        (0, 255, 0),
        max(2, round(min(frame.shape[:2]) / 500)),
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"failed to write detection preview: {output_path}")


def render_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
    info: VideoInfo,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="retinal-mask-",
            suffix=".png",
            dir=output_path.parent,
            delete=False,
        ) as temporary_mask:
            mask_path = Path(temporary_mask.name)
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError("failed to write temporary circular mask")

        filter_graph = (
            "[0:v]format=rgba[video];"
            "[1:v]format=gray[mask];"
            "[video][mask]alphamerge[masked];"
            f"color=c=black:s={info.width}x{info.height}:r={info.fps:.12g}[black];"
            "[black][masked]overlay=shortest=1:format=auto,"
            "format=yuv420p[out]"
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(input_path),
            "-loop",
            "1",
            "-framerate",
            f"{info.fps:.12g}",
            "-i",
            str(mask_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with status {result.returncode}")
    finally:
        if mask_path is not None:
            mask_path.unlink(missing_ok=True)


def render_with_opencv(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
    info: VideoInfo,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("could not open the OpenCV video reader or writer")

    mask_float = (mask.astype(np.float32) / 255.0)[:, :, None]
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(np.round(frame.astype(np.float32) * mask_float).astype(np.uint8))
    finally:
        capture.release()
        writer.release()


def render_tracking_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    initial_circle: Circle,
    info: VideoInfo,
    min_radius_fraction: float,
    max_radius_fraction: float,
    margin: float,
    feather: float,
    smoothing: float,
    max_center_shift: float,
    max_radius_change: float,
    detect_every: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    """Track each frame and stream masked BGR frames directly into FFmpeg."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{info.width}x{info.height}",
        "-framerate",
        f"{info.fps:.12g}",
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen input video: {input_path}")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    current = initial_circle
    accepted_detections = 0
    reused_detections = 0
    frame_index = 0
    try:
        if process.stdin is None:
            raise RuntimeError("could not open the FFmpeg input pipe")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % detect_every == 0:
                current, accepted = track_retinal_circle(
                    frame,
                    current,
                    min_radius_fraction=min_radius_fraction,
                    max_radius_fraction=max_radius_fraction,
                    margin=margin,
                    smoothing=smoothing,
                    max_center_shift=max_center_shift,
                    max_radius_change=max_radius_change,
                )
                if accepted:
                    accepted_detections += 1
                else:
                    reused_detections += 1
            masked = apply_circular_mask(frame, current, feather)
            try:
                process.stdin.write(masked.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg stopped while receiving tracked frames") from exc
            frame_index += 1
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
    finally:
        capture.release()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    print(
        f"Tracked {frame_index} frame(s): {accepted_detections} detection(s) "
        f"accepted, {reused_detections} fallback(s)."
    )


def render_tracking_with_opencv(
    input_path: Path,
    output_path: Path,
    initial_circle: Circle,
    info: VideoInfo,
    min_radius_fraction: float,
    max_radius_fraction: float,
    margin: float,
    feather: float,
    smoothing: float,
    max_center_shift: float,
    max_radius_change: float,
    detect_every: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("could not open the OpenCV video reader or writer")

    current = initial_circle
    accepted_detections = 0
    reused_detections = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % detect_every == 0:
                current, accepted = track_retinal_circle(
                    frame,
                    current,
                    min_radius_fraction=min_radius_fraction,
                    max_radius_fraction=max_radius_fraction,
                    margin=margin,
                    smoothing=smoothing,
                    max_center_shift=max_center_shift,
                    max_radius_change=max_radius_change,
                )
                if accepted:
                    accepted_detections += 1
                else:
                    reused_detections += 1
            writer.write(apply_circular_mask(frame, current, feather))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    print(
        f"Tracked {frame_index} frame(s): {accepted_detections} detection(s) "
        f"accepted, {reused_detections} fallback(s)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the circular retinal aperture in an MP4 or MOV and turn every "
            "pixel outside it black."
        )
    )
    parser.add_argument("input", type=Path, help="Input .mp4 or .mov video.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .mp4 (default: INPUT_retina_only.mp4).",
    )
    parser.add_argument("--center-x", type=float, help="Manual circle center X in pixels.")
    parser.add_argument("--center-y", type=float, help="Manual circle center Y in pixels.")
    parser.add_argument("--radius", type=non_negative_float, help="Manual radius in pixels.")
    parser.add_argument(
        "--margin",
        type=non_negative_float,
        default=0.0,
        help="Move the detected edge inward by this many pixels.",
    )
    parser.add_argument(
        "--feather",
        type=non_negative_float,
        default=0.0,
        help="Width in pixels of an optional soft inner edge.",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=9,
        help="Frames used for automatic detection.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=non_negative_float,
        default=8.0,
        help="Sample frames within this many seconds from the start.",
    )
    parser.add_argument(
        "--min-radius-fraction",
        type=fraction,
        default=0.16,
        help="Smallest automatic radius as a fraction of the shorter frame side.",
    )
    parser.add_argument(
        "--max-radius-fraction",
        type=fraction,
        default=0.36,
        help="Largest automatic radius as a fraction of the shorter frame side.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Write one image with the detected circle drawn in green.",
    )
    parser.add_argument(
        "--static-mask",
        action="store_true",
        help="Use one circle for the whole video instead of detecting every frame.",
    )
    parser.add_argument(
        "--detect-every",
        type=positive_int,
        default=1,
        help="Run tracking detection every N frames (default: every frame).",
    )
    parser.add_argument(
        "--smoothing",
        type=smoothing_factor,
        default=0.45,
        help="Tracking response from 0 to 1; higher follows motion faster.",
    )
    parser.add_argument(
        "--max-center-shift",
        type=fraction,
        default=0.05,
        help="Largest accepted center movement per detection, as a frame fraction.",
    )
    parser.add_argument(
        "--max-radius-change",
        type=fraction,
        default=0.03,
        help="Largest accepted radius change per detection, as a frame fraction.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="FFmpeg preserves audio; OpenCV is the dependency-only fallback.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        choices=range(0, 52),
        default=18,
        metavar="[0-51]",
        help="FFmpeg H.264 quality (lower is higher quality).",
    )
    parser.add_argument(
        "--preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        default="medium",
        help="FFmpeg H.264 speed/size preset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_retina_only.mp4")
    )

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"input video does not exist: {input_path}")
        validate_video_extensions(input_path, output_path)
        if input_path == output_path:
            raise ValueError("input and output paths must be different")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"output already exists: {output_path} (use --overwrite to replace it)"
            )

        manual_values = (args.center_x, args.center_y, args.radius)
        if any(value is not None for value in manual_values) and not all(
            value is not None for value in manual_values
        ):
            raise ValueError("--center-x, --center-y, and --radius must be used together")
        if args.min_radius_fraction >= args.max_radius_fraction:
            raise ValueError(
                "--min-radius-fraction must be smaller than --max-radius-fraction"
            )

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open input video: {input_path}")
        try:
            info = read_video_info(capture, input_path)
            samples = sample_video_frames(
                capture,
                info,
                sample_count=args.samples,
                sample_seconds=args.sample_seconds,
            )
        finally:
            capture.release()

        if all(value is not None for value in manual_values):
            circle = Circle(args.center_x, args.center_y, args.radius)
            mode = "manual"
        else:
            circle = detect_retinal_circle(
                samples,
                min_radius_fraction=args.min_radius_fraction,
                max_radius_fraction=args.max_radius_fraction,
            )
            mode = "automatic"

        circle = Circle(circle.center_x, circle.center_y, circle.radius - args.margin)
        validate_circle(circle, info.width, info.height)
        print(
            f"{mode.capitalize()} retinal view: center=({circle.center_x:.1f}, "
            f"{circle.center_y:.1f}), radius={circle.radius:.1f}px"
        )
        if args.static_mask:
            print("Using one static mask for the whole video.")
        else:
            print(f"Tracking the retinal view every {args.detect_every} frame(s).")

        if args.preview is not None:
            write_preview(samples[len(samples) // 2], circle, args.preview)
            print(f"Wrote detection preview: {args.preview}")

        ffmpeg_available = shutil.which("ffmpeg") is not None
        use_ffmpeg = args.backend == "ffmpeg" or (
            args.backend == "auto" and ffmpeg_available
        )
        if args.backend == "ffmpeg" and not ffmpeg_available:
            raise RuntimeError("FFmpeg was requested but is not installed")

        short_side = min(info.width, info.height)
        max_center_shift = args.max_center_shift * short_side
        max_radius_change = args.max_radius_change * short_side
        if args.static_mask:
            mask = make_circular_mask(
                info.width,
                info.height,
                circle,
                feather=args.feather,
            )
            if use_ffmpeg:
                render_with_ffmpeg(
                    input_path,
                    output_path,
                    mask,
                    info,
                    crf=args.crf,
                    preset=args.preset,
                    overwrite=args.overwrite,
                )
            else:
                print(
                    "FFmpeg is unavailable; using OpenCV output without audio.",
                    file=sys.stderr,
                )
                render_with_opencv(
                    input_path,
                    output_path,
                    mask,
                    info,
                    overwrite=args.overwrite,
                )
        elif use_ffmpeg:
            render_tracking_with_ffmpeg(
                input_path,
                output_path,
                circle,
                info,
                min_radius_fraction=args.min_radius_fraction,
                max_radius_fraction=args.max_radius_fraction,
                margin=args.margin,
                feather=args.feather,
                smoothing=args.smoothing,
                max_center_shift=max_center_shift,
                max_radius_change=max_radius_change,
                detect_every=args.detect_every,
                crf=args.crf,
                preset=args.preset,
                overwrite=args.overwrite,
            )
        else:
            print(
                "FFmpeg is unavailable; using OpenCV output without audio.",
                file=sys.stderr,
            )
            render_tracking_with_opencv(
                input_path,
                output_path,
                circle,
                info,
                min_radius_fraction=args.min_radius_fraction,
                max_radius_fraction=args.max_radius_fraction,
                margin=args.margin,
                feather=args.feather,
                smoothing=args.smoothing,
                max_center_shift=max_center_shift,
                max_radius_change=max_radius_change,
                detect_every=args.detect_every,
                overwrite=args.overwrite,
            )
        print(f"Wrote masked video: {output_path}")
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
