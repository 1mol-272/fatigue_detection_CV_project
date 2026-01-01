"""
Landmark extraction (flat-module version, no Python package required).

Based on the original provided `extract_landmarks.py` logic:
- Recursively scan videos under input_root
- Mirror the input folder structure under output_root
- For each video: MediaPipe FaceMesh -> (T,468,3) landmarks + frame_indices
- Save: *_landmarks.npz and *_meta.json

Outputs:
  <output_root>/<relative_subdir>/<video_stem>_landmarks.npz
  <output_root>/<relative_subdir>/<video_stem>_meta.json
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp


EXPECTED_LANDMARKS = 468
VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v")


def list_videos(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXTS):
                yield os.path.join(dirpath, fn)


def make_output_paths(video_path: str, input_root: str, output_root: str):
    rel = os.path.relpath(video_path, input_root)
    rel_dir = os.path.dirname(rel)
    base = os.path.splitext(os.path.basename(video_path))[0]

    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_npz = os.path.join(out_dir, f"{base}_landmarks.npz")
    out_meta = os.path.join(out_dir, f"{base}_meta.json")
    return out_npz, out_meta


def safe_landmark_array(lm_list):
    arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
    if arr.shape == (EXPECTED_LANDMARKS, 3):
        return arr

    fixed = np.full((EXPECTED_LANDMARKS, 3), np.nan, dtype=np.float32)
    m = min(arr.shape[0], EXPECTED_LANDMARKS)
    fixed[:m, :] = arr[:m, :]
    return fixed


def infer_label_from_path(video_path: str, label_rules=None):
    """
    label_rules: list of (pattern, label), if pattern appears in path -> label (case-insensitive)
    """
    if not label_rules:
        return None
    s = video_path.lower()
    for pattern, label in label_rules:
        if pattern.lower() in s:
            return label
    return None


def extract_one_video(
    video_path: str,
    process_every_n_frames: int = 1,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    label_rules=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face_mesh = mp.solutions.face_mesh

    landmarks_list = []
    frame_indices = []

    face_detected_frames = 0
    processed_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            if process_every_n_frames > 1 and (frame_id % process_every_n_frames != 0):
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                arr = safe_landmark_array(lm)
                face_detected_frames += 1
            else:
                arr = np.full((EXPECTED_LANDMARKS, 3), np.nan, dtype=np.float32)

            landmarks_list.append(arr)
            frame_indices.append(frame_id)
            processed_frames += 1

            if processed_frames % 300 == 0:
                print(f"  processed {processed_frames} frames...")

    cap.release()

    if landmarks_list:
        landmarks = np.stack(landmarks_list, axis=0).astype(np.float32)
    else:
        landmarks = np.empty((0, EXPECTED_LANDMARKS, 3), dtype=np.float32)

    frame_indices = np.array(frame_indices, dtype=np.int32)

    label = infer_label_from_path(video_path, label_rules)

    meta = {
        "video_path": video_path,
        "fps": float(fps) if fps is not None else None,
        "total_frames_est": total_frames_est,
        "processed_frames": int(landmarks.shape[0]),
        "process_every_n_frames": int(process_every_n_frames),
        "face_detected_frames": int(face_detected_frames),
        "face_detected_ratio": float(face_detected_frames / max(1, landmarks.shape[0])),
        "landmarks_shape": list(landmarks.shape),
        "coords": "mediapipe_normalized_xyz",
        "label": label,
        "note": "Frames with no detected face contain NaN landmarks.",
    }

    return landmarks, frame_indices, meta


def run_extract_landmarks(
    input_root: str = "videos",
    output_root: str = "data",
    process_every_n_frames: int = 1,
    skip_existing: bool = True,
    label_rules=None,
):
    os.makedirs(output_root, exist_ok=True)

    for video_path in list_videos(input_root):
        out_npz, out_meta = make_output_paths(video_path, input_root, output_root)

        if skip_existing and os.path.exists(out_npz) and os.path.exists(out_meta):
            print(f"SKIP (exists): {video_path}")
            continue

        print(f"\nExtracting: {video_path}")
        landmarks, frame_indices, meta = extract_one_video(
            video_path=video_path,
            process_every_n_frames=process_every_n_frames,
            label_rules=label_rules,
        )

        np.savez_compressed(out_npz, landmarks=landmarks, frame_indices=frame_indices)
        with open(out_meta, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        print(f"Saved: {out_npz}")
        print(f"Saved: {out_meta}")
