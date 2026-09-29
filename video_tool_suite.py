# -*- coding: utf-8 -*-
"""
视频工具套件（镜头拆分 + 情绪标注 一体化图形界面版）
========================================================

把两个独立脚本组合成带图形界面（GUI）的单程序：

  阶段A  镜头自动拆分   （来自 shot_splitter.py）
         用 PySceneDetect 检测转场，OpenCV 人脸检测保证镜头首尾人脸完整，
         ffmpeg 按镜头切分。

  阶段B  主情绪标注     （来自 openface3_video_emotion.py）
         用 OpenFace 3.0 逐帧统计视频（或每个镜头）的主情绪，
         并按选择把情绪写入文件名（改名 / 复制 / 仅预览）。

界面说明
--------
- 顶部菜单“帮助 -> 参数说明”可随时查看所有参数的详细说明（帮助文档）。
- 任务按“输入视频 -> 阶段A拆分 -> 阶段B情绪标注”的顺序自动执行。
- 运行过程在“日志”页实时显示；重型依赖（cv2 / scenedetect / torch /
  openface）在点击“开始”时才按需加载，缺失时给出中文提示，不影响界面启动。

运行方式
--------
    python video_tool_suite.py

所需环境（与原始脚本一致）
--------------------------
    pip install opencv-python scenedetect[opencv] imageio-ffmpeg
    pip install torch openface  # 阶段B 需要；并下载 openface 权重(见帮助)

图形界面使用标准库 tkinter，无需额外安装。
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# PyInstaller 打包后 torch/numpy 的 DLL 被解压到 sys._MEIPASS 子目录，
# Windows 默认不会去子目录找依赖，需手动加入 DLL 搜索路径，否则导入 torch
# 可能报 WinError 1114。仅在 frozen(exe) 模式下执行。
# 注意：add_dll_directory 返回的句柄必须保持引用，否则会被垃圾回收随即
# 把该目录从搜索路径移除（CPython 官方文档明确警告）。
_FROZEN_DLL_HANDLES = []
if getattr(sys, "frozen", False) and sys.platform == "win32":
    _meip = getattr(sys, "_MEIPASS", "")
    for _d in ("torch/lib", "numpy/.libs"):
        _p = os.path.normpath(os.path.join(_meip, _d))
        if os.path.isdir(_p):
            try:
                _FROZEN_DLL_HANDLES.append(os.add_dll_directory(_p))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 依赖按需加载：返回模块，缺失时抛清晰中文错误
# ---------------------------------------------------------------------------
def need(module, hint=""):
    try:
        return __import__(module)
    except Exception as exc:
        raise RuntimeError(
            f"缺少依赖模块 {module}，无法执行该步骤。\n请先安装：pip install {module}"
            + (f"\n补充说明：{hint}" if hint else "")
            + f"\n原始错误：{exc}"
        )


# ---------------------------------------------------------------------------
# 共享：定位资源、ffmpeg
# ---------------------------------------------------------------------------
def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(name):
    candidates = [os.path.join(app_dir(), name)]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(os.path.join(bundle_dir, name))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def find_ffmpeg(cli_path=None, log=None):
    def say(msg):
        if log:
            log(msg)

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = [cli_path,
                  os.environ.get("SHOT_SPLITTER_FFMPEG"),
                  os.environ.get("FFMPEG_PATH"),
                  os.path.join(app_dir(), exe_name),
                  os.path.join(app_dir(), "ffmpeg", exe_name)]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(os.path.join(bundle_dir, "ffmpeg", exe_name))
        candidates.append(os.path.join(bundle_dir, exe_name))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "找不到 ffmpeg：请把 ffmpeg.exe 放到程序同目录、加入系统 PATH，"
        "或在界面中指定完整路径。"
    )


# ===========================================================================
# 阶段A：镜头拆分
# ===========================================================================
class FaceChecker:
    """人脸完整性检查。优先 OpenCV YuNet 模型，缺失时回退 Haar 级联。"""

    YUNET_MODEL = resource_path("face_detection_yunet_2023mar.onnx")

    def __init__(self, margin_ratio=0.05, min_face_ratio=0.02, log=None):
        self.margin_ratio = margin_ratio
        self.min_face_ratio = min_face_ratio
        cv2 = need("cv2")
        self.yunet = None
        if os.path.isfile(self.YUNET_MODEL) and hasattr(cv2, "FaceDetectorYN"):
            self.yunet = self._load_yunet()
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self.detector = (cv2.CascadeClassifier(cascade_path)
                         if os.path.isfile(cascade_path) else None)
        if self.yunet is None and self.detector is None:
            if log:
                log(f"  警告：YuNet 模型与 Haar 级联都不可用，人脸检测将始终返回空。")

    def _load_yunet(self):
        cv2 = need("cv2")
        try:
            with open(self.YUNET_MODEL, "rb") as model_file:
                model_magic = model_file.read(16)
            if not model_magic:
                raise ValueError("模型文件为空")
            if model_magic.lstrip().lower().startswith(b"<!doctype html"):
                raise ValueError("模型文件是 HTML 错误页，不是 ONNX 模型")
            try:
                return cv2.FaceDetectorYN.create(
                    self.YUNET_MODEL, "", (320, 320), score_threshold=0.6
                )
            except cv2.error:
                if not self.YUNET_MODEL.isascii():
                    return self._load_yunet_from_ascii_path()
                raise
        except Exception as exc:
            raise RuntimeError(f"YuNet 模型不可用：{exc}")

    def _load_yunet_from_ascii_path(self):
        cv2 = need("cv2")
        model_dir = tempfile.mkdtemp(prefix="shot_splitter_model_")
        model_path = os.path.join(model_dir, "face_detection_yunet.onnx")
        with open(self.YUNET_MODEL, "rb") as source, open(model_path, "wb") as target:
            target.write(source.read())
        return cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=0.6
        )

    def detect(self, frame):
        cv2 = need("cv2")
        if self.yunet is not None:
            h, w = frame.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is None:
                return []
            return [(int(fx), int(fy), int(fw), int(fh))
                    for fx, fy, fw, fh in faces[:, :4]]
        if self.detector is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    def truncated_faces(self, frame):
        h, w = frame.shape[:2]
        mx, my = w * self.margin_ratio, h * self.margin_ratio
        min_side = min(w, h) * self.min_face_ratio
        bad = []
        for (x, y, fw, fh) in self.detect(frame):
            if fw < min_side and fh < min_side:
                continue
            if x <= mx or y <= my or x + fw >= w - mx or y + fh >= h - my:
                bad.append((x, y, fw, fh))
        return bad

    def complete_faces(self, frame):
        h, w = frame.shape[:2]
        mx, my = w * self.margin_ratio, h * self.margin_ratio
        min_side = min(w, h) * self.min_face_ratio
        good = []
        for (x, y, fw, fh) in self.detect(frame):
            if fw < min_side and fh < min_side:
                continue
            if x > mx and y > my and x + fw < w - mx and y + fh < h - my:
                good.append((x, y, fw, fh))
        return good


def detect_shots(video_path, threshold=27.0, min_scene_len=15, log=None):
    """检测镜头边界，返回 [(start_frame, end_frame), ...] 与 fps。"""
    need("scenedetect")
    from scenedetect import ContentDetector, SceneManager, open_video
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list()
    fps = float(video.frame_rate)
    total = video.duration.frame_num
    if not scenes:
        scenes = [(video.base_timecode, video.duration)]
    shots = [(s.frame_num, e.frame_num) for s, e in scenes]
    if shots[-1][1] < total:
        shots[-1] = (shots[-1][0], total)
    return shots, fps


def refine_cuts(video_path, shots, checker, window=10, log=None):
    """在每个转场点附近 ±window 帧内重选切割帧，避开被截断的人脸。"""
    cv2 = need("cv2")
    cap = cv2.VideoCapture(video_path)
    refined = list(shots)

    def frame_ok(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return False
        return len(checker.truncated_faces(frame)) == 0

    for i in range(1, len(shots)):
        cut = shots[i][0]
        best = None
        for offset in range(0, window + 1):
            for cand in (cut + offset, cut - offset):
                lo = shots[i - 1][0] + 2
                hi = shots[i][1] - 2
                if cand <= lo or cand >= hi:
                    continue
                if frame_ok(cand - 1) and frame_ok(cand):
                    best = cand
                    break
            if best is not None:
                break
        if best is not None and best != cut:
            refined[i - 1] = (refined[i - 1][0], best)
            refined[i] = (best, refined[i][1])
    cap.release()
    return refined


def shot_face_stats(video_path, start, end, checker, samples=8):
    cv2 = need("cv2")
    cap = cv2.VideoCapture(video_path)
    length = max(end - start, 1)
    step = max(length // samples, 1)
    good, total = 0, 0
    idx = start
    while idx < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        if checker.complete_faces(frame):
            good += 1
        idx += step
    cap.release()
    return good, total


def split_video(video_path, shots, fps, out_dir, fast=False, vcodec="libx264",
                crf=18, ffmpeg="ffmpeg", log=None):
    """用 ffmpeg 按镜头切分，返回输出文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    outputs = []
    for i, (start, end) in enumerate(shots, 1):
        t0 = start / fps
        dur = (end - start) / fps
        out_path = os.path.join(out_dir, f"{base}_shot{i:03d}.mp4")
        if fast:
            cmd = [ffmpeg, "-y", "-ss", f"{t0:.3f}", "-i", video_path,
                   "-t", f"{dur:.3f}", "-c", "copy", out_path]
        else:
            cmd = [ffmpeg, "-y", "-i", video_path,
                   "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
                   "-c:v", vcodec, "-crf", str(crf), "-preset", "fast",
                   "-c:a", "aac", out_path]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outputs.append(out_path)
        if log:
            log(f"    [{i}/{len(shots)}] {os.path.basename(out_path)}  "
                f"{t0:.2f}s -> {t0 + dur:.2f}s")
    return outputs


def run_stage_a(video_path, out_dir, p, ffmpeg, log):
    """执行阶段A：检测 -> 合并短镜头 -> 优化切点 -> 校验 -> 切分。返回切出的文件列表与报告。"""
    if not os.path.isfile(video_path):
        raise RuntimeError(f"找不到输入文件: {video_path}")

    log("== 阶段A 1/4 检测镜头边界 (PySceneDetect) ==")
    min_frames = max(int(p["min_len"] * 30), 1)
    shots, fps = detect_shots(video_path, p["threshold"], min_frames)
    log(f"   检测到 {len(shots)} 个镜头, fps={fps:.3f}")

    merged = []
    for s in shots:
        if merged and (s[1] - s[0]) < p["min_len"] * fps:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    shots = merged

    checker = FaceChecker(log=log)
    log("== 阶段A 2/4 优化切割点，避开被截断的人脸 ==")
    shots = refine_cuts(video_path, shots, checker, window=p["refine_window"])

    log("== 阶段A 3/4 校验每个镜头内的人脸完整性 ==")
    report = []
    kept_shots = []
    for i, (start, end) in enumerate(shots, 1):
        good, total = shot_face_stats(video_path, start, end, checker)
        has_face = good > 0
        report.append({
            "shot": i, "start_frame": start, "end_frame": end,
            "start_time": round(start / fps, 3), "end_time": round(end / fps, 3),
            "duration": round((end - start) / fps, 3),
            "frames_with_complete_face": good, "sampled_frames": total,
            "has_complete_face": has_face,
        })
        tag = "OK" if has_face else "无完整人脸"
        log(f"   镜头 {i:3d}: {start / fps:8.2f}s - {end / fps:8.2f}s  完整人脸帧 {good}/{total} [{tag}]")
        if has_face or not p["require_face"]:
            kept_shots.append((start, end))

    log(f"== 阶段A 4/4 切分视频 ({'流拷贝' if p['fast'] else '帧精确重编码'}) ==")
    outputs = split_video(video_path, kept_shots, fps, out_dir,
                          fast=p["fast"], crf=p["crf"], ffmpeg=ffmpeg, log=log)

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"input": os.path.abspath(video_path), "fps": fps,
                   "shots": report}, f, ensure_ascii=False, indent=2)
    log(f"  完成：{len(outputs)} 个短视频已保存到 {out_dir}")
    log(f"  检测报告：{report_path}")
    if p["require_face"] and len(kept_shots) < len(shots):
        log(f"  已跳过 {len(shots) - len(kept_shots)} 个无完整人脸的镜头")
    return outputs, report_path


# ===========================================================================
# 阶段B：OpenFace 主情绪标注
# ===========================================================================
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise",
                  "Fear", "Disgust", "Anger", "Contempt"]


def resolve_weights_dir(explicit):
    if explicit:
        return os.path.abspath(explicit)
    side = os.path.join(app_dir(), "weights")
    if os.path.isdir(side):
        return side
    bundle = os.path.join(getattr(sys, "_MEIPASS", ""), "weights")
    if os.path.isdir(bundle):
        return bundle
    return "weights"


def detect_top_face_on_frame(detector, frame_bgr):
    """对单帧做人脸检测，返回置信度最高的人脸裁剪（BGR）。"""
    need("torch")
    import numpy as np
    import torch
    from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
    from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
    from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    from openface.Pytorch_Retinaface.data import cfg_mnet
    cfg_mnet["pretrain"] = False

    img = np.float32(frame_bgr)
    im_height, im_width, _ = img.shape
    inp = img.copy()
    inp -= (104, 117, 123)
    inp = inp.transpose(2, 0, 1)
    inp = torch.from_numpy(inp).unsqueeze(0).to(detector.device)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
                         ).to(detector.device)
    with torch.no_grad():
        loc, conf, landms = detector.model(inp)
    priorbox = PriorBox(detector.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(detector.device)
    boxes = decode(loc.data.squeeze(0), priors.data, detector.cfg["variance"])
    boxes = (boxes * scale).cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    inds = np.where(scores > detector.confidence_threshold)[0]
    boxes, scores = boxes[inds], scores[inds]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, detector.nms_threshold)
    dets = dets[keep]
    if dets is None or len(dets) == 0:
        return None, None
    det = dets[0]
    if det[4] < detector.vis_threshold:
        return None, None
    bbox = det[:4].astype(int)
    x1, y1, x2, y2 = bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_bgr.shape[1]), min(y2, frame_bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None, None
    face = frame_bgr[y1:y2, x1:x2]
    return face, det


def ensure_openface_retina():
    """确认 openface 包完整可用（含 Pytorch_Retinaface 子包），缺失时给出针对性指引。"""
    import importlib.util
    spec = importlib.util.find_spec("openface")
    if spec is None:
        raise RuntimeError(
            "未安装 openface，无法执行阶段B情绪分析。\n请先安装：pip install openface，"
            "并运行 openface download 下载权重。")
    try:
        import openface.Pytorch_Retinaface  # noqa: F401
    except Exception:
        loc = getattr(spec, "submodule_search_locations", None)
        loc_str = ""
        if loc:
            first = list(loc)[0]
            loc_str = f"\n检测到 openface 包位于：{first}。"
        raise RuntimeError(
            "当前环境中的 openface 包缺少 Pytorch_Retinaface 子模块，无法执行情绪分析。\n"
            "可能原因：装成了旧版或不完整的 openface。\n"
            "解决办法：改用包含完整 openface 的环境（如 E:\\cut\\.venv_of3），"
            "并在其中补装 scenedetect 以便阶段A运行：\n"
            "    E:\\cut\\.venv_of3\\Scripts\\pip install \"scenedetect[opencv]\"\n"
            "然后使用该环境的 python 启动本程序即可两个阶段同时工作。" + loc_str)


def load_openface(device, weights_dir, log):
    """加载 OpenFace 检测器与多任务预测器，返回 (face_detector, predictor)。"""
    need("torch", "OpenFace 需要 torch（CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu）")
    ensure_openface_retina()
    # RetinaFace 默认会从当前目录相对路径加载 ./weights/mobilenetV1X0.25_pretrain.tar
    # （Alignment_RetinaFace.pth 已含完整权重，该 tar 是冗余的），关掉以解耦运行目录。
    from openface.Pytorch_Retinaface.data import cfg_mnet
    cfg_mnet["pretrain"] = False
    from openface.face_detection import FaceDetector
    from openface.multitask_model import MultitaskPredictor
    face_w = os.path.join(resolve_weights_dir(weights_dir), "Alignment_RetinaFace.pth")
    mtl_w = os.path.join(resolve_weights_dir(weights_dir), "MTL_backbone.pth")
    for p in (face_w, mtl_w):
        if not os.path.exists(p):
            raise RuntimeError(
                f"缺少权重文件: {p}\n请先运行: openface download --output {resolve_weights_dir(weights_dir)}")
    if log:
        log("  加载 OpenFace 模型 ...")
    face_detector = FaceDetector(model_path=face_w, device=device)
    predictor = MultitaskPredictor(model_path=mtl_w, device=device)
    return face_detector, predictor


def analyze_video_emotion(video_path, device, step, weights_dir, log):
    """分析视频主情绪，返回 (dominant_label, stats)。"""
    if not os.path.exists(video_path):
        raise RuntimeError(f"视频不存在: {video_path}")
    cv2 = need("cv2")
    import numpy as np
    import torch
    ensure_openface_retina()
    from openface.Pytorch_Retinaface.data import cfg_mnet
    cfg_mnet["pretrain"] = False

    face_detector, predictor = load_openface(device, weights_dir, log)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if log:
        log(f"  视频共约 {total_frames} 帧, 采样步长 {step}")

    votes = Counter()
    prob_sum = {}
    face_frames = 0
    sampled = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            sampled += 1
            face, _ = detect_top_face_on_frame(face_detector, frame)
            if face is not None and face.size > 0:
                face_frames += 1
                emotion_logits, _, _ = predictor.predict(face)
                prob = torch.softmax(emotion_logits, dim=1)[0].detach().cpu().numpy()
                label = int(np.argmax(prob))
                votes[label] += 1
                prob_sum.setdefault(label, 0.0)
                prob_sum[label] += float(prob[label])
            if sampled % 50 == 0 and log:
                log(f"    已处理 {sampled} 帧 ...")
        frame_idx += 1
    cap.release()

    if face_frames == 0:
        raise RuntimeError("未在任何采样帧中检测到人脸，无法计算主情绪。")

    if log:
        log("  ==== 各情绪得票（帧数）====")
        for i, lbl in enumerate(EMOTION_LABELS):
            if votes.get(i):
                log(f"    {lbl:<10} {votes[i]} 帧  平均置信度 {prob_sum[i]/votes[i]:.3f}")

    best_count = max(votes.values())
    top = [i for i, c in votes.items() if c == best_count]
    if len(top) == 1:
        dominant = top[0]
    else:
        dominant = max(top, key=lambda i: prob_sum[i] / votes[i])
    dom_label = EMOTION_LABELS[dominant]
    if log:
        log(f"  主情绪: {dom_label}  (得票 {best_count}/{face_frames} 帧, "
            f"占比 {best_count/face_frames*100:.1f}%)")
    stats = {"label": dom_label, "votes": dict(votes),
             "face_frames": face_frames, "best_count": best_count}
    return dom_label, stats


def apply_emotion_to_file(video_path, dom_label, action, log):
    """按 action 处理文件名：rename 改名 / copy 复制 / dry 仅预览。"""
    stem, ext = os.path.splitext(video_path)
    new_path = f"{stem}_{dom_label}{ext}"
    if action == "dry":
        if log:
            log(f"    [仅预览] 将标注为: {os.path.basename(new_path)}")
        return None
    if action == "copy":
        if os.path.exists(new_path):
            if log:
                log(f"    [跳过] 目标已存在，未复制: {os.path.basename(new_path)}")
            return new_path
        shutil.copy2(video_path, new_path)
        if log:
            log(f"    已复制: {os.path.basename(video_path)} -> {os.path.basename(new_path)}")
        return new_path
    # rename
    if os.path.exists(new_path):
        if log:
            log(f"    [跳过] 目标已存在，未覆盖: {os.path.basename(new_path)}")
        return new_path
    os.rename(video_path, new_path)
    if log:
        log(f"    已改名: {os.path.basename(video_path)} -> {os.path.basename(new_path)}")
    return new_path


def run_stage_b(targets, device, step, weights_dir, action, log):
    """对一组视频逐个计算主情绪并标注文件名，返回结果列表。"""
    log("== 阶段B 主情绪标注 (OpenFace 3.0) ==")
    results = []
    for idx, target in enumerate(targets, 1):
        log(f"  [{idx}/{len(targets)}] {os.path.basename(target)}")
        dom_label, stats = analyze_video_emotion(target, device, step, weights_dir, log)
        new_path = apply_emotion_to_file(target, dom_label, action, log)
        results.append({"file": target, "label": dom_label,
                         "new_file": new_path, "stats": stats})
    return results


# ===========================================================================
# 图形界面
# ===========================================================================
class App:
    HELP_TEXT = """\
视频工具套件 —— 参数说明（帮助文档）

一、任务设置（阶段开关）
  输入视频：      单个视频文件，或一个文件夹（批量处理其中所有视频）。
  输出目录：      阶段A拆分出的镜头保存位置；缺省为“视频同目录_镜头”，
                  文件夹批量时为“文件夹名_镜头”并逐个视频建子目录。
  ffmpeg 路径：   可选。留空时按 程序目录 -> 内置 -> 系统PATH 自动查找。

二、阶段A：镜头拆分参数
  ★ 阈值 threshold（默认 27.0，越小越灵敏）
      转场检测灵敏度。画面内容变化超过阈值即判定为新镜头。
      值越小，越容易把细微变化也当成镜头切换，镜头数越多；
      值越大，只在大幅转场时才切分，镜头更少更长。范围建议 15~40。
  ★ 最短镜头时长 min-len（默认 0.5 秒）
      短于该时长的镜头会被并入相邻镜头，避免产生碎片片段。
  ★ 切点优化范围 refine-window（默认 10 帧）
      在每个转场点前后 ±N 帧内搜索“首尾帧人脸完整”的切割位置，
      用于避开人脸被画面边缘截断的情况。越大搜索越充分但越慢。
  ★ 仅保留含完整人脸镜头 require-face（默认 关闭）
      勾选后，无人脸完整出现的镜头将被跳过，不参与拆分与输出，
      并在 report.json 中记录；不勾选则全部镜头都切出。
  ★ 快速流拷贝 fast（默认 关闭）
      勾选后使用 ffmpeg 流拷贝（-c copy），速度快但只能在关键帧处
      切分，边界可能偏移数帧，人脸保障也因此失效。追求精确请保持关闭。
  ★ 重编码质量 crf（默认 18）
      帧精确模式下的视频质量参数，越小画质越高、文件越大（常用 18~23）。

三、阶段B：主情绪标注参数
  ★ 推理设备 device（cpu / cuda）
      无独立显卡用 cpu；有 NVIDIA 显卡且装好 CUDA 版 torch 用 cuda 加速。
  ★ 采样步长 step（默认 1，每 N 帧采样一帧）
      每 N 帧采样一帧做情绪统计，加快处理。值越大越快、精度略降。
      处理长视频建议 2~5。
  ★ 权重目录 weights（可留空自动查找）
      含 Alignment_RetinaFace.pth、MTL_backbone.pth 的目录。
      查找顺序：界面指定 -> 程序同目录 weights -> 打包内置 -> 当前目录。
  ★ 处理动作 action（改名 / 复制 / 仅预览）
      - 改名：把文件重命名为“原名_情绪.mp4”（不保留原文件）；
      - 复制：生成“原名_情绪.mp4”新文件（保留原文件）；
      - 仅预览：只显示结果，不生成或修改任何文件。

四、八类情绪（OpenFace 3.0 AffectNet 标签）
  Neutral(中性) Happy(开心) Sad(悲伤) Surprise(惊讶)
  Fear(恐惧)  Disgust(厌恶) Anger(愤怒) Contempt(轻蔑)

五、主情绪判定规则
  整段视频中“出现帧数最多”的情绪为多数票主情绪；
  若出现并列，则取并列情绪中“平均置信度更高”者。

六、组合流程
  默认顺序：输入视频 -> 阶段A拆分出镜头 -> 阶段B对每个镜头标注情绪。
  也可只开阶段A（只拆分）、或只开阶段B（对原视频直接标注情绪）。
  输入为文件夹时：自动逐个处理其中所有视频(mp4/avi/mov/mkv/flv/wmv)，
  每个视频的镜头输出到各自子目录；单个视频失败不影响其他视频。

七、环境依赖
  pip install opencv-python scenedetect[opencv] imageio-ffmpeg
  pip install torch openface   # 阶段B需要；另需 openface download 下载权重
  界面本身只用标准库 tkinter，缺依赖时点“开始”会给出中文提示。
"""

    def __init__(self, root):
        self.root = root
        root.title("视频工具套件 : 镜头拆分 + 主情绪标注")
        root.geometry("860x480")
        self.msg_queue = queue.Queue()
        self.running = False

        self.build_menu()
        self.build_layout()
        self.root.after(120, self._drain_queue)

    # ---- 菜单 ----
    def build_menu(self):
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="参数说明（帮助文档）", command=self.show_help)
        help_menu.add_separator()
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

    # ---- 主布局：上方设置页签 + 中部控制行 + 底部日志 ----
    def build_layout(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        self.tab_task = ttk.Frame(nb); nb.add(self.tab_task, text="① 任务设置")
        self.tab_split = ttk.Frame(nb); nb.add(self.tab_split, text="② 拆分参数(阶段A)")
        self.tab_emo = ttk.Frame(nb); nb.add(self.tab_emo, text="③ 情绪参数(阶段B)")

        self._build_task_tab()
        self._build_split_tab()
        self._build_emo_tab()

        # 控制行：开始按钮 + 进度条（在日志上方）
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill="x", padx=8, pady=6)
        self.btn_run = ttk.Button(self.control_frame, text="▶ 开始处理", command=self.start)
        self.btn_run.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(self.control_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=6)

        # 底部日志：实时滚动
        self._build_log_pane()

    # ---- ① 任务设置 ----
    def _build_task_tab(self):
        f = self.tab_task
        pad = dict(padx=10, pady=6, sticky="w")

        tk.Label(f, text="输入视频(文件/文件夹)：").grid(row=0, column=0, **pad)
        self.var_input = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_input, width=60).grid(row=0, column=1, columnspan=2, **pad)
        ttk.Button(f, text="选文件", command=self._pick_input).grid(row=0, column=3, **pad)
        ttk.Button(f, text="选文件夹(批量)", command=self._pick_input_dir).grid(row=0, column=4, **pad)

        tk.Label(f, text="输出目录：").grid(row=1, column=0, **pad)
        self.var_out = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_out, width=70).grid(row=1, column=1, columnspan=2, **pad)
        ttk.Button(f, text="浏览...", command=self._pick_out).grid(row=1, column=3, **pad)

        tk.Label(f, text="ffmpeg 路径：").grid(row=2, column=0, **pad)
        self.var_ffmpeg = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_ffmpeg, width=70).grid(row=2, column=1, columnspan=2, **pad)
        ttk.Button(f, text="浏览...", command=self._pick_ffmpeg).grid(row=2, column=3, **pad)

        ttk.Separator(f, orient="horizontal").grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=8)

        self.var_use_a = tk.BooleanVar(value=True)
        self.var_use_b = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="启用 阶段A：镜头拆分", variable=self.var_use_a).grid(row=4, column=0, columnspan=2, **pad)
        ttk.Checkbutton(f, text="启用 阶段B：主情绪标注", variable=self.var_use_b).grid(row=4, column=2, columnspan=2, **pad)
        tk.Label(f, text="（默认顺序：先拆分出镜头，再对每个镜头标注情绪）",
                 fg="#666").grid(row=5, column=0, columnspan=4, padx=10, sticky="w")

    def _pick_input(self):
        p = filedialog.askopenfilename(title="选择单个输入视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("所有文件", "*.*")])
        if p:
            self.var_input.set(p)
            self._auto_out(p)

    def _pick_input_dir(self):
        p = filedialog.askdirectory(title="选择文件夹（批量处理其中所有视频）")
        if p:
            self.var_input.set(p)
            self._auto_out(p)

    def _auto_out(self, path):
        if self.var_out.get():
            return
        if os.path.isdir(path):
            base = os.path.basename(os.path.normpath(path))
            self.var_out.set(os.path.join(os.path.dirname(os.path.normpath(path)), base + "_镜头"))
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            self.var_out.set(os.path.join(os.path.dirname(path), stem + "_镜头"))

    def _pick_out(self):
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.var_out.set(p)

    def _pick_ffmpeg(self):
        p = filedialog.askopenfilename(title="选择 ffmpeg",
            filetypes=[("ffmpeg", "ffmpeg.exe ffmpeg"), ("所有文件", "*.*")])
        if p:
            self.var_ffmpeg.set(p)

    # ---- ② 拆分参数 ----
    def _build_split_tab(self):
        f = self.tab_split
        pad = dict(padx=10, pady=6, sticky="w")

        def row(r, label, var, tip):
            tk.Label(f, text=label).grid(row=r, column=0, **pad)
            ttk.Entry(f, textvariable=var, width=16).grid(row=r, column=1, **pad)
            tk.Label(f, text=tip, fg="#666").grid(row=r, column=2, sticky="w", padx=6)

        self.var_threshold = tk.StringVar(value="27.0")
        self.var_minlen = tk.StringVar(value="0.5")
        self.var_refine = tk.StringVar(value="10")
        self.var_crf = tk.StringVar(value="18")

        row(0, "转场阈值 threshold", self.var_threshold, "越小越灵敏，建议 15~40")
        row(1, "最短镜头时长 min-len(s)", self.var_minlen, "过短镜头并入邻居")
        row(2, "切点优化范围 refine-window(帧)", self.var_refine, "±N 帧内搜索人脸完整切点")
        row(3, "重编码质量 crf", self.var_crf, "越小画质越高，常用 18~23")

        self.var_require_face = tk.BooleanVar(value=True)
        self.var_fast = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="仅保留含完整人脸的镜头（require-face）",
                        variable=self.var_require_face).grid(row=4, column=0, columnspan=2, padx=10, pady=6, sticky="w")
        ttk.Checkbutton(f, text="快速流拷贝（fast，切点不精确、人脸保障失效）",
                        variable=self.var_fast).grid(row=5, column=0, columnspan=2, padx=10, pady=6, sticky="w")

    # ---- ③ 情绪参数 ----
    def _build_emo_tab(self):
        f = self.tab_emo
        pad = dict(padx=10, pady=6, sticky="w")

        tk.Label(f, text="推理设备 device：").grid(row=0, column=0, **pad)
        self.var_device = tk.StringVar(value="cpu")
        ttk.Combobox(f, textvariable=self.var_device, values=["cpu", "cuda"],
                     state="readonly", width=8).grid(row=0, column=1, sticky="w", padx=10, pady=6)

        tk.Label(f, text="采样步长 step（每 N 帧取一帧）：").grid(row=1, column=0, **pad)
        self.var_step = tk.StringVar(value="1")
        ttk.Entry(f, textvariable=self.var_step, width=16).grid(row=1, column=1, sticky="w", padx=10, pady=6)
        tk.Label(f, text="越大越快、精度略降；长视频建议 2~5", fg="#666").grid(row=1, column=2, sticky="w", padx=6)

        tk.Label(f, text="权重目录 weights：").grid(row=2, column=0, **pad)
        self.var_weights = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_weights, width=56).grid(row=2, column=1, sticky="w", padx=10, pady=6)
        ttk.Button(f, text="浏览...", command=self._pick_weights).grid(row=2, column=2, sticky="w", padx=6)

        tk.Label(f, text="处理动作：").grid(row=3, column=0, **pad)
        self.var_action = tk.StringVar(value="rename")
        actions = [("改名（不保留原文件）", "rename"),
                   ("复制为新文件（保留原文件）", "copy"),
                   ("仅预览（不改动文件）", "dry")]
        for i, (txt, val) in enumerate(actions):
            ttk.Radiobutton(f, text=txt, variable=self.var_action,
                            value=val).grid(row=4 + i, column=1, sticky="w", padx=10)

        tk.Label(f, text="八类情绪：Neutral 中性 / Happy 开心 / Sad 悲伤 / Surprise 惊讶 / "
                         "Fear 恐惧 / Disgust 厌恶 / Anger 愤怒 / Contempt 轻蔑",
                 fg="#666").grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="w")

    def _pick_weights(self):
        p = filedialog.askdirectory(title="选择 OpenFace 权重目录")
        if p:
            self.var_weights.set(p)

    # ---- 底部日志（实时滚动，自动滚到底部） ----
    def _build_log_pane(self):
        self.log_frame = ttk.LabelFrame(self.root, text="运行日志（实时滚动）")
        self.log_frame.pack(fill="both", expand=False, padx=8, pady=(0, 6))
        self.log_text = tk.Text(self.log_frame, wrap="none", height=14, state="disabled")
        sb = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    # ---- 帮助文档 ----
    def show_help(self):
        win = tk.Toplevel(self.root)
        win.title("帮助文档 —— 参数说明")
        win.geometry("760x620")
        txt = tk.Text(win, wrap="word")
        sb = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", self.HELP_TEXT)
        txt.config(state="disabled")

    def show_about(self):
        messagebox.showinfo(
            "关于",
            "视频工具套件\n"
            "镜头拆分(shot_splitter) + 主情绪标注(openface3_video_emotion) 一体化版\n"
            "帮助 -> 参数说明 可查看全部参数说明。")

    # ---- 日志写入（线程安全） ----
    def log(self, msg):
        self.msg_queue.put(str(msg))

    def _drain_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    # ---- 启动任务 ----
    def _read_float(self, var, name, default):
        try:
            return float(var.get())
        except ValueError:
            raise RuntimeError(f"参数 {name} 不是有效数字: {var.get()}")

    def _read_int(self, var, name, default):
        try:
            return int(var.get())
        except ValueError:
            raise RuntimeError(f"参数 {name} 不是有效整数: {var.get()}")

    def start(self):
        if self.running:
            self.log("任务已在运行中，请等待完成。")
            return
        input_video = self.var_input.get().strip()
        if not input_video:
            messagebox.showerror("错误", "请先选择输入视频（单个视频或文件夹）。")
            return
        if not (os.path.isfile(input_video) or os.path.isdir(input_video)):
            messagebox.showerror("错误", f"输入不存在：{input_video}")
            return
        input_is_dir = os.path.isdir(input_video)
        if input_is_dir:
            exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
            found = [n for n in os.listdir(input_video)
                     if os.path.isfile(os.path.join(input_video, n)) and n.lower().endswith(exts)]
            if not found:
                messagebox.showerror("错误", f"文件夹中没有找到视频文件：{input_video}")
                return
        if input_is_dir:
            base = os.path.basename(os.path.normpath(input_video))
            default_out = os.path.join(os.path.dirname(os.path.normpath(input_video)), base + "_镜头")
        else:
            default_out = os.path.splitext(input_video)[0] + "_镜头"
        out_dir = self.var_out.get().strip() or default_out
        ffmpeg = self.var_ffmpeg.get().strip() or None

        params = {
            "use_a": self.var_use_a.get(),
            "use_b": self.var_use_b.get(),
            "a": {
                "threshold": self._read_float(self.var_threshold, "threshold", 27.0),
                "min_len": self._read_float(self.var_minlen, "min-len", 0.5),
                "refine_window": self._read_int(self.var_refine, "refine-window", 10),
                "crf": self._read_int(self.var_crf, "crf", 18),
                "require_face": self.var_require_face.get(),
                "fast": self.var_fast.get(),
            },
            "b": {
                "device": self.var_device.get(),
                "step": self._read_int(self.var_step, "step", 1),
                "weights": self.var_weights.get().strip(),
                "action": self.var_action.get(),
            },
            "input": input_video,
            "input_is_dir": input_is_dir,
            "out_dir": out_dir,
            "ffmpeg": ffmpeg,
        }
        self.running = True
        self.btn_run.config(state="disabled")
        self.progress.start(12)
        self.log("==============================================")
        self.log(f"任务开始  输入: {input_video}")
        self.log(f"阶段开关  拆分={'是' if params['use_a'] else '否'}  情绪={'是' if params['use_b'] else '否'}")
        threading.Thread(target=self._worker, args=(params,), daemon=True).start()

    def _collect_inputs(self, path):
        # 返回待处理视频列表：单个文件 -> [文件]；文件夹 -> 其中所有视频
        if os.path.isdir(path):
            exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
            return sorted(
                os.path.join(path, n) for n in os.listdir(path)
                if os.path.isfile(os.path.join(path, n)) and n.lower().endswith(exts))
        return [path]

    def _worker(self, p):
        try:
            if not p["use_a"] and not p["use_b"]:
                self.log("两个阶段都未启用，没有可执行的任务。")
                return
            videos = self._collect_inputs(p["input"])
            if not videos:
                self.log("未找到可处理的视频文件。")
                return
            self.log(f"共发现 {len(videos)} 个待处理视频。")
            for i, video in enumerate(videos, 1):
                self.log(f"===== [{i}/{len(videos)}] 处理视频: {os.path.basename(video)} =====")
                self._process_one(video, i, len(videos), p)
            self.log(">>> 全部完成。")
        finally:
            self.root.after(0, self._finish)

    def _process_one(self, video, i, total, p):
        try:
            # 文件夹输入时，每个视频单独一个输出子目录
            if p.get("input_is_dir"):
                stem = os.path.splitext(os.path.basename(video))[0]
                out_dir = os.path.join(p["out_dir"], stem)
            else:
                out_dir = p["out_dir"]
            targets = []
            if p["use_a"]:
                targets, _ = run_stage_a(video, out_dir, p["a"],
                                         find_ffmpeg(p["ffmpeg"], self.log), self.log)
            if p["use_b"]:
                if p["use_a"]:
                    targets = [t for t in targets if os.path.isfile(t)]
                    if not targets:
                        self.log("  阶段A未产出镜头文件，跳过阶段B。")
                    else:
                        self.log(f"  阶段B 将对阶段A拆分的 {len(targets)} 个镜头全部进行主情绪标注。")
                else:
                    targets = [video]
                run_stage_b(targets, p["b"]["device"], p["b"]["step"],
                            p["b"]["weights"], p["b"]["action"], self.log)
        except Exception as exc:
            self.log(f"!!! 处理 {os.path.basename(video)} 出错：{exc}")
        else:
            self.log(f"--- 视频 [{i}/{total}] {os.path.basename(video)} 处理完成。")

    def _finish(self):
        self.running = False
        self.btn_run.config(state="normal")
        self.progress.stop()
        self.progress.config(value=0)


def selftest_main():
    """隐藏自检模式：video_tool_suite.exe --selftest <视频路径>
    验证打包后的导入链与两个阶段能否工作，结果写入 exe 同目录
    video_tool_suite_selftest.txt（GUI 窗口不弹出）。"""
    import importlib.util as _iu
    args = [a for a in sys.argv if not a.startswith("--")]
    video = args[1] if len(args) > 1 else None
    out_path = os.path.join(app_dir(), "video_tool_suite_selftest.txt")
    lines = []

    def log(m):
        lines.append(str(m))

    try:
        log("=== video_tool_suite --selftest ===")
        for m in ("cv2", "scenedetect", "torch", "openface",
                  "openface.Pytorch_Retinaface"):
            log(f"import {m}: {_iu.find_spec(m) is not None}")
        if not video:
            raise RuntimeError("用法: video_tool_suite.exe --selftest <视频路径>")
        if not os.path.isfile(video):
            raise RuntimeError(f"视频不存在: {video}")

        log("阶段A: 检测镜头边界 ...")
        shots, fps = detect_shots(video, 27.0, 15, log)
        log(f"  镜头数: {len(shots)}, fps={fps:.3f}")

        log("阶段B: 主情绪分析 ...")
        label, stats = analyze_video_emotion(video, "cpu", 10, None, log)
        log(f"  主情绪: {label}")
        log(f"  统计: {stats}")
        log("SELFTEST OK")
    except Exception as exc:
        log(f"SELFTEST FAIL: {type(exc).__name__}: {exc}")
    finally:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass
        sys.exit(0)


def main():
    if "--selftest" in sys.argv:
        return selftest_main()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
