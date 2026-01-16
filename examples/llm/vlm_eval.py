# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def invoke_vlm_infer_test(args):
    cmd = [
        args.vlm_infer_test_path, "--libInferPath", args.libvlm_infer_path,
        "--modelType", args.model_type, "--llmEnginePath", args.llm_engine_dir,
        "--visualEnginePath", args.vit_engine_dir, "--inputFile",
        args.input_json, "--outputFile", args.output_json, "--staticImageSize",
        "--staticPrompt"
    ]
    if args.isEagle3:
        cmd.append("--eagle3")

    print("执行命令：")
    print(" ".join(cmd))

    try:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1)

        # 实时打印输出
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"vlm_infer_test 执行失败，返回码: {process.returncode}")

        print("vlm_infer_test 执行完成")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"vlm_infer_test 不存在或不可执行: {args.vlm_infer_test_path}")


def generate_input_json(image_dir, user_prompt, input_json_path):
    """
    image_dir: 包含图片的目录（支持 JPG/JPEG/PNG）
    user_prompt: user 的文本 prompt（字符串）
    input_json_path: 输出 json 文件路径
    """

    # ================= 可变参数（集中在这里，方便修改） =================
    BATCH_SIZE = 1
    TEMPERATURE = 0.0
    TOP_P = 1.0
    TOP_K = 1
    MAX_GENERATE_LENGTH = 128

    SYSTEM_PROMPT = "You are a helpful assistant."

    # ====================================================================

    requests = []

    image_files = sorted(f for f in os.listdir(image_dir)
                         if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS)

    if not image_files:
        raise ValueError("image dir 中未找到支持的图片格式")

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)

        request = {
            "messages": [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role":
                "user",
                "content": [{
                    "type": "image",
                    "image": img_path
                }, {
                    "type": "text",
                    "text": user_prompt
                }]
            }]
        }

        requests.append(request)

    final_json = {
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_generate_length": MAX_GENERATE_LENGTH,
        "requests": requests
    }

    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)


def combine_images_cv2(far_img, wide_img, save_path):
    # 读取两张图片
    up_img = cv2.imread(far_img)
    down_img = cv2.imread(wide_img)

    print(
        f"ModelInference[CV2] far img size {up_img.shape}, wide img size {down_img.shape}"
    )
    # 处理第一张图片
    if up_img.shape[1] == 3840 and up_img.shape[0] == 2160:
        up_img = cv2.resize(up_img, (960, 540), interpolation=cv2.INTER_LINEAR)
        far_box = (0, 126 + 28, 952, 400 + 28)
    elif up_img.shape[1] == 960 and up_img.shape[0] == 540:
        far_box = (0, 126 + 28, 952, 400 + 28)
    elif up_img.shape[1] == 960 and up_img.shape[0] == 512:
        far_box = (0, 126, 952, 400)
    else:
        print(f"ModelInference[CV2] invalid img size")
        return False

    # 处理第二张图片
    if down_img.shape[1] == 3840 and down_img.shape[0] == 2160:
        down_img = cv2.resize(down_img, (960, 540),
                              interpolation=cv2.INTER_LINEAR)
        wide_box = (0, 160, 952, 390)
    elif down_img.shape[1] == 960 and down_img.shape[0] == 540:
        wide_box = (0, 160, 952, 390)
    elif down_img.shape[1] == 960 and down_img.shape[0] == 512:
        wide_box = (0, 160, 952, 390)
    else:
        print(f"ModelInference[CV2] invalid img size")
        return False

    # 裁剪图片
    up_img_cropped = up_img[far_box[1]:far_box[3], far_box[0]:far_box[2]]
    down_img_cropped = down_img[wide_box[1]:wide_box[3],
                                wide_box[0]:wide_box[2]]

    # 创建新画布 (高度=504, 宽度=952, 3通道BGR)
    combined_img = np.ones((504, 952, 3), dtype=np.uint8) * 255  # 白色背景

    # 粘贴第一张图片到顶部
    h1, w1 = up_img_cropped.shape[:2]
    combined_img[0:h1, 0:w1] = up_img_cropped

    # 粘贴第二张图片到底部 (从y=274开始)
    h2, w2 = down_img_cropped.shape[:2]
    combined_img[274:274 + h2, 0:w2] = down_img_cropped

    # 保存图片
    cv2.imwrite(save_path, combined_img)
    print(
        f"ModelInference[CV2] {far_img} {wide_img} combine save to {save_path}"
    )
    return True


def combine_wide_and_far(wide_image_dir, far_image_dir, model_type,
                         output_image_dir):
    if model_type == "qwen2_vl":
        print("combine images for qwen2_vl")
    else:
        print("unsupported model type for combine image.")
        raise RuntimeError("unsupported model type for combine image.")

    # 遍历far文件夹
    far_files = {
        f: os.path.join(far_image_dir, f)
        for f in os.listdir(far_image_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS
    }

    # 遍历wide文件夹
    wide_files = {
        f: os.path.join(wide_image_dir, f)
        for f in os.listdir(wide_image_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS
    }

    # 获取所有文件名的并集
    all_files = set(far_files.keys()).union(wide_files.keys())

    paired = []
    for fname in sorted(all_files):
        far_path = far_files.get(fname, "")
        wide_path = wide_files.get(fname, "")
        if far_path != "" and wide_path != "":
            paired.append((far_path, wide_path))

    def clear_img_files(folder_path):
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        for ext in extensions:
            pattern = os.path.join(folder_path, ext)
            for img_file in glob.glob(pattern):
                os.remove(img_file)

    clear_img_files(output_image_dir)

    for far_img, wide_img in paired:
        combine_images_cv2(far_img,
                           wide_img,
                           save_path=os.path.join(
                               output_image_dir,
                               "combined_" + os.path.basename(wide_img)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM inference parameter parser")

    # ----------- 输入源 -----------
    parser.add_argument("--input_json",
                        type=str,
                        required=True,
                        help="input json 文件路径")

    parser.add_argument("--image_dirs",
                        type=str,
                        nargs="+",
                        default=None,
                        help="一个或两个图像文件夹路径")

    # ----------- prompt（可选）-----------
    parser.add_argument("--system_prompt",
                        type=str,
                        default="You are a helpful assistant.",
                        help="system prompt")

    parser.add_argument(
        "--user_prompt",
        type=str,
        default=
        "<image>图片为相同时刻下，Far与Wide相机拍摄到的场景拼接照片。沿y轴方向[0,274]为far相机图像，[274,504]为Wide相机图像。根据图像，描述当前场景光照情况与道路状况，特别关注是 否存在施工区域与占道情况，并给出合理化驾驶建议",
        help="user prompt")

    # ----------- 输出 -----------
    parser.add_argument("--output_json",
                        type=str,
                        required=True,
                        help="输出 json 路径")

    # ----------- VLM 推理相关（可选）-----------
    parser.add_argument("--vlm_infer_test_path",
                        type=str,
                        default="./build/examples/llm/vlm_infer_test",
                        help="vlm_infer_test 可执行文件路径")

    parser.add_argument("--libvlm_infer_path",
                        type=str,
                        default="./build/examples/llm/libvlm_infer.so",
                        help="libvlm_infer.so 路径")

    parser.add_argument("--plugin_path",
                        type=str,
                        default="./build",
                        help="plugin 路径")

    # ----------- 模型相关（必填）-----------
    parser.add_argument("--model_type", type=str, required=True, help="模型类型")

    parser.add_argument("--llm_engine_dir",
                        type=str,
                        required=True,
                        help="LLM engine 目录")

    parser.add_argument("--vit_engine_dir",
                        type=str,
                        required=True,
                        help="ViT engine 目录")

    parser.add_argument("--isEagle3",
                        action="store_true",
                        help="是否为 Eagle3 模型")

    args = parser.parse_args()
    return args


def validate_args(args):
    # input_json 和 image_dirs 必须至少给一个
    if not os.path.exists(args.input_json) and args.image_dirs is None:
        raise ValueError("input_json 路径不存在时，需要指定 image_dirs 用于生成 input.json")

    # image_dirs 最多只能两个
    if args.image_dirs is not None and len(args.image_dirs) > 2:
        raise ValueError("image_dirs 最多只能包含两个目录")

    # 如果两个都给了，以 input_json 为准
    if args.input_json is not None and args.image_dirs is not None:
        print("[INFO] 同时提供了 input_json 和 image_dirs，将以 input_json 为准")

    if args.image_dirs is not None:
        for d in args.image_dirs:
            if not Path(d).exists():
                raise FileNotFoundError(f"image_dir 不存在: {d}")

    support_models = ["qwen2_vl", "qwen3_vl"]
    if args.model_type is not None and args.model_type not in support_models:
        raise ValueError(f"Only support model type {support_models}")


def main():
    args = parse_args()
    validate_args(args)

    # 统一决定输入源
    input_source = "json" if args.input_json is not None else "image_dirs"

    print("===== 参数解析结果 =====")
    print(f"input_source      : {input_source}")
    print(f"input_json        : {args.input_json}")
    print(f"image_dirs        : {args.image_dirs}")
    print(f"system_prompt     : {args.system_prompt}")
    print(f"user_prompt       : {args.user_prompt}")
    print(f"output_json       : {args.output_json}")
    print(f"vlm_infer_test    : {args.vlm_infer_test_path}")
    print(f"libvlm_infer.so   : {args.libvlm_infer_path}")
    print(f"plugin_path       : {args.plugin_path}")
    print(f"model_type        : {args.model_type}")
    print(f"llm_engine_dir    : {args.llm_engine_dir}")
    print(f"vit_engine_dir    : {args.vit_engine_dir}")
    print(f"isEagle3          : {args.isEagle3}")

    TEMP_COMBINE_IMG_DIR = "./temp"
    if not os.path.exists(args.input_json):
        if len(args.image_dirs) == 2:
            dst_img_dir = TEMP_COMBINE_IMG_DIR
            combine_wide_and_far(args.image_dirs[0], args.image_dirs[1],
                                 args.model_type, dst_img_dir)
        else:
            dst_img_dir = args.image_dirs[0]

        generate_input_json(dst_img_dir, args.user_prompt, args.input_json)

    invoke_vlm_infer_test(args)


if __name__ == "__main__":
    main()
