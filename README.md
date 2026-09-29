# 视频工具套件（镜头拆分 + 主情绪标注）

把两个脚本 **shot\_splitter.py**（镜头拆分）与 **openface3\_video\_emotion.py**（主情绪标注）

组合为一个带图形界面（GUI）的程序，并支持参数调节与内置帮助文档。

## 打包版（免安装环境，可直接在其它电脑运行）

`dist\video_tool_suite\` 已打包成可移植 exe：内置 Python 运行时、torch、openface
（含 Pytorch_Retinaface）、cv2、scenedetect、ffmpeg、OpenFace 权重与 tkinter，
目标电脑无需安装 Python / ffmpeg / torch 等任何环境。

```
dist\video_tool_suite\video_tool_suite.exe      # 双击即可打开图形界面
```

把整个 `dist\video_tool_suite` 文件夹拷贝到任意 64 位 Windows（Win10/11）电脑即可运行。

## 源码运行

推荐用已验证可用的环境 `E:\cut\.venv_of3`（同时具备两个阶段全部依赖）：

```
E:\cut\.venv_of3\Scripts\python.exe video\_tool\_suite.py
```

该环境已含 opencv-python / scenedetect / torch / openface（含 Pytorch_Retinaface），
OpenFace 权重已随程序放在 `weights\` 子目录，程序会自动定位。

> 若报错 `No module named 'openface.Pytorch_Retinaface'`，说明用了不含完整 openface 的环境，
> 请改用 `.venv_of3` 启动。

界面本身只依赖标准库 `tkinter`；重型依赖（cv2 /scenedetect/torch /openface）

在点击「开始」时才按需加载，缺失时给出中文提示，不影响界面启动。

## 所需环境



```
pip install opencv-python scenedetect\[opencv] imageio-ffmpeg

pip install torch openface      # 阶段B需要；并运行 openface download 下载权重
```

程序会自动查找 ffmpeg（程序目录 / 内置 / 系统 PATH）；找不到时可在界面指定路径。

## 界面结构

设置区（页签）：

| 页签            | 作用                            |
| ------------- | ----------------------------- |
| ① 任务设置        | 输入视频、输出目录、ffmpeg 路径、阶段开关 |
| ② 拆分参数 (阶段 A) | 镜头拆分的全部可调参数                   |
| ③ 情绪参数 (阶段 B) | 主情绪标注的全部可调参数                  |

页签下方：**开始处理** 按钮 + **进度条**；
窗口底部：**运行日志**（实时滚动，自动滚到底部）。

顶部菜单「帮助 → 参数说明」可随时查看完整帮助文档（即本文件）。

## 组合流程

默认：`输入视频 → 阶段A 拆分出镜头 → 阶段B 对每个镜头标注主情绪`。

也可只开阶段 A（只拆分），或只开阶段 B（对原视频直接标注情绪）。

**批量处理**：输入为文件夹时，自动逐个处理其中所有视频（mp4 /avi/mov/mkv/flv/wmv），
每个视频的镜头输出到各自子目录；单个视频失败不影响其他视频。



***

## 参数说明

### 一、任务设置



| 参数        | 说明                               |
| --------- | -------------------------------- |
| 输入视频      | 单个视频文件，或一个文件夹（批量处理其中所有视频）   |
| 输出目录      | 镜头保存位置；留空默认 “视频同目录\_镜头”；文件夹批量时 “文件夹名\_镜头” 并为每个视频建子目录 |
| ffmpeg 路径 | 可选。留空按 程序目录 → 内置 → 系统 PATH 自动查找  |
| 阶段开关      | 勾选是否启用阶段 A / 阶段 B                |

### 二、阶段 A：镜头拆分参数

**阈值 threshold（默认 27.0，越小越灵敏）**

转场检测灵敏度，画面内容变化超过阈值即判定为新镜头。值越小，越容易把细微变化当成镜头切换、镜头数越多；值越大，只在大幅转场时才切分。建议 15\~40。

**最短镜头时长 min-len（默认 0.5 秒）**

短于该时长的镜头并入相邻镜头，避免碎片片段。

**切点优化范围 refine-window（默认 10 帧）**

在每个转场点前后 ±N 帧内搜索 “首尾帧人脸完整” 的切割位置，避开人脸被画面边缘截断的情况。越大搜索越充分但越慢。

**仅保留含完整人脸镜头 require-face（默认关闭）**

勾选后，无人脸完整出现的镜头被跳过（记入 report.json）；不勾选则全部切出。

**快速流拷贝 fast（默认关闭）**

勾选后使用 ffmpeg 流拷贝（-c copy），速度快但只能在关键帧处切分，边界可能偏移、人脸保障失效。追求精确请保持关闭。

**重编码质量 crf（默认 18）**

帧精确模式下的视频质量，越小画质越高、文件越大（常用 18\~23）。

### 三、阶段 B：主情绪标注参数

**推理设备 device（cpu /cuda）**

无独立显卡用 cpu；有 NVIDIA 显卡且装好 CUDA 版 torch 用 cuda 加速。

**采样步长 step（默认 1，每 N 帧取一帧）**

每 N 帧采样一帧做情绪统计，越大越快、精度略降。长视频建议 2\~5。

**权重目录 weights（可留空自动查找）**

含 `Alignment_RetinaFace.pth`、`MTL_backbone.pth` 的目录。查找顺序：界面指定 → 程序同目录 weights → 打包内置 → 当前目录。

**处理动作 action（改名 / 复制 / 仅预览）**



* 改名：重命名为 “原名\_情绪.mp4”（不保留原文件）；

* 复制：生成 “原名\_情绪.mp4” 新文件（保留原文件）；

* 仅预览：只显示结果，不生成或修改任何文件。

### 四、八类情绪（OpenFace 3.0 AffectNet 标签）

Neutral 中性・Happy 开心・Sad 悲伤・Surprise 惊讶・

Fear 恐惧・Disgust 厌恶・Anger 愤怒・Contempt 轻蔑

### 五、主情绪判定规则

整段视频中 “出现帧数最多” 的情绪为多数票主情绪；出现并列时取并列情绪中平均置信度更高者。

## 说明



* 阶段 B 处理的是单张主脸（每帧取置信度最高的人脸）。

* 阶段 A 会输出 `report.json` 检测报告（镜头起止、时长、人脸完整性等）。