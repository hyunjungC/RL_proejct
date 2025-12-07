'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

'''
conda activate voca_render
cd /workspace/RL-VOCASET_mjh

# 무음 모드(강화학습 적용)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2_mjh/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2_mjh/results_render \
  --fps 30 \
  --background_black

# 무음 모드(only_stage1)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2/results_render \
  --fps 30 \
  --background_black
  
# 무음 모드(loss_RL)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalker/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalker/results_render \
  --fps 30 \
  --background_black

# 무음 모드(gt_render(test))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/test \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render \
  --fps 30 \
  --background_black
# 무음 모드(gt_render(val))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/val \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render \
  --fps 30 \
  --background_black
# 무음 모드(gt_render(train))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/train \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render \
  --fps 30 \
  --background_black


################################################################################

# 유음 모드(강화학습 적용)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2_mjh/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2_mjh/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

# 유음 모드(only_stage1)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalk_v2/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio
  
# 유음 모드(loss_RL)
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/checkpoints/selftalker/results \
  --output /workspace/RL-VOCASET_mjh/checkpoints/selftalker/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

# 유음 모드(gt_render(test))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/test \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render_audio \
  --fps 30 \
  --background_black \
  --with_audio
# 유음 모드(gt_render(val))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/val \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render_audio \
  --fps 30 \
  --background_black \
  --with_audio
  # 유음 모드(gt_render(train))
python /workspace/RL-VOCASET_mjh/02.render_mjh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_mjh/vocaset/results/gt/npy/train \
  --output /workspace/RL-VOCASET_mjh/vocaset/results/gt/gt_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

  '''




# ===============================================================================
# RL-VOCASET_my 
# ==============================================================================
# RL + baseline (FaceFormer_hj) 
# ==============================================================================
"""
python /workspace/RL-VOCASET_my copy/render_jh copy.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_my copy/checkpoints/faceformer_hj_nostyle/styledependant/results \
  --output /workspace/RL-VOCASET_my copy/checkpoints/faceformer_hj_nostyle/styledependant/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

"""
# ==============================================================================
# Only baseline (FaceFormer_hj) 
# ==============================================================================
"""
python /workspace/RL-VOCASET_my/render_jh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_my/checkpoints/faceformer_base/styleIndependant/results\
  --output /workspace/RL-VOCASET_my/checkpoints/faceformer_base/styleIndependant/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

"""

# ==============================================================================
# Only baseline (FaceFormer_hj) 
# ==============================================================================
"""
python /workspace/RL-VOCASET_my/render_jh.py \
  --dataset vocaset \
  --render_template_path templates \
  --pred_path /workspace/RL-VOCASET_my/vocaset/vertices_npy\
  --output /workspace/RL-VOCASET_my/vocaset/vertices_npy/results_render_audio \
  --fps 30 \
  --background_black \
  --with_audio

"""


import os
import cv2
import tempfile
import numpy as np
from subprocess import call
import argparse

# OpenGL을 headless(EGL)로 사용
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import trimesh


def render_mesh_helper(args, vertices, faces, t_center, rot=None, tex_img=None, z_offset=0.0):
    """
    vertices: (V, 3)
    faces   : (F, 3)
    """
    if rot is None:
        rot = np.zeros(3, dtype=np.float32)
    else:
        rot = np.asarray(rot, dtype=np.float32)

    # 카메라 파라미터 (CodeTalker/VOCA 스타일)
    if args.dataset.lower() == "biwi":
        camera_params = {
            "c": np.array([400, 400]),
            "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            "f": np.array([4754.97941935 / 8, 4754.97941935 / 8]),
        }
        
    else:  # vocaset
        camera_params = {
            "c": np.array([400, 400]),
            "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
        }

    frustum = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}

    # vertex 회전/정렬
    verts = np.asarray(vertices, dtype=np.float32).copy()
    t_center = np.asarray(t_center, dtype=np.float32)
    R, _ = cv2.Rodrigues(rot)
    verts = (R.dot((verts - t_center).T)).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8,
    )

    tri_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=rgb_per_v,
        process=False,
    )
    render_mesh = pyrender.Mesh.from_trimesh(
        tri_mesh, material=primitive_material, smooth=True
    )

    if args.background_black:
        scene = pyrender.Scene(
            ambient_light=[0.2, 0.2, 0.2],
            bg_color=[0, 0, 0],
        )
    else:
        scene = pyrender.Scene(
            ambient_light=[0.2, 0.2, 0.2],
            bg_color=[255, 255, 255],
        )

    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["f"][0],
        fy=camera_params["f"][1],
        cx=camera_params["c"][0],
        cy=camera_params["c"][1],
        znear=frustum["near"],
        zfar=frustum["far"],
    )

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset], dtype=np.float32)
    scene.add(
        camera,
        pose=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
    )

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1.0, 1.0, 1.0])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(
            viewport_width=frustum["width"],
            viewport_height=frustum["height"],
        )
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print("pyrender: Failed rendering frame:", e)
        color = np.zeros((frustum["height"], frustum["width"], 3), dtype="uint8")

    return color[..., ::-1]


def render_sequence_meshes(args, sequence_vertices, faces, out_path, predicted_vertices_path):
    """
    sequence_vertices: (T, V, 3)
    faces           : (F, 3)
    """
    num_frames = sequence_vertices.shape[0]
    file_name_pred = os.path.basename(predicted_vertices_path).split(".")[0]

    tmp_video_file_pred = tempfile.NamedTemporaryFile(
        "w", suffix=".mp4", dir=out_path
    )
    writer_pred = cv2.VideoWriter(
        tmp_video_file_pred.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (800, 800),
        True,
    )

    center = np.mean(sequence_vertices[0], axis=0)
    video_fname_pred = os.path.join(out_path, file_name_pred + ".mp4")

    # --------- 프레임 렌더링 ----------
    for i_frame in range(num_frames):
        frame_verts = sequence_vertices[i_frame]
        pred_img = render_mesh_helper(
            args,
            frame_verts,
            faces,
            center,
            tex_img=None,
        )
        pred_img = pred_img.astype(np.uint8)
        writer_pred.write(pred_img)

    writer_pred.release()

    # 1단계: 무음 비디오(yuv420p) 인코딩 (두 모드 공통)
    cmd = (
        f"ffmpeg -y -loglevel error "
        f"-i {tmp_video_file_pred.name} "
        f"-pix_fmt yuv420p -qscale 0 {video_fname_pred}"
    ).split()
    call(cmd)

    # 2단계: with_audio 플래그가 켜져 있을 때만 오디오 붙이기
    if args.with_audio:
        wav_path = os.path.join(args.wav_dir, file_name_pred + ".wav")

        # FaceFormer 스타일 파일명에는 "_condition_XXX"가 붙어서 원본 wav와 이름이 달라질 수 있다.
        # 그런 경우 condition 앞부분만 떼서 wav를 찾는다.
        if not os.path.exists(wav_path) and "_condition_" in file_name_pred:
            stripped = file_name_pred.split("_condition_")[0]
            alt_wav_path = os.path.join(args.wav_dir, stripped + ".wav")
            if os.path.exists(alt_wav_path):
                print(f"[INFO] wav fallback: {os.path.basename(file_name_pred)} -> {os.path.basename(alt_wav_path)}")
                wav_path = alt_wav_path

        if os.path.exists(wav_path):
            video_with_audio = video_fname_pred.replace(".mp4", "_audio.mp4")

            cmd = (
                f"ffmpeg -y -loglevel error "
                f"-i {wav_path} -i {video_fname_pred} "
                f"-vcodec h264 -ac 2 -channel_layout stereo -qscale 0 "
                f"{video_with_audio}"
            ).split()
            call(cmd)

            # 원본 무음 비디오는 지워도 됨
            if os.path.exists(video_fname_pred):
                os.remove(video_fname_pred)

            print(f"[AUDIO] {os.path.basename(wav_path)} → {os.path.basename(video_with_audio)}")
        else:
            print(f"[WARN] wav not found for {file_name_pred}: {wav_path}")
    else:
        print(f"[MUTE] saved silent video: {video_fname_pred}")


def main():
    parser = argparse.ArgumentParser(
        description="Render npy (vertices) into mp4 videos (psbody-free)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vocaset",
        help="vocaset or BIWI",
    )
    parser.add_argument(
        "--render_template_path",
        type=str,
        default="templates",
        help="path of the mesh in FLAME/BIWI topology",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="folder containing *.npy prediction files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path of the rendered video sequences (base dir)",
    )
    parser.add_argument(
        "--background_black",
        action="store_true",
        help="use black background (else white)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frame rate - 30 for vocaset; 25 for BIWI",
    )
    parser.add_argument(
        "--vertice_dim",
        type=int,
        default=5023 * 3,
        help="number of vertices - 5023*3 for vocaset; 23370*3 for BIWI",
    )
    parser.add_argument(
        "--with_audio",
        action="store_true",
        help="Set this flag to attach wav audio (if found) to the video.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="Directory where wav files are stored (default: <dataset>/wav)",
    )

    args = parser.parse_args()

    # BIWI일 때 기본 fps / vertice_dim 보정 (필요하면)
    if args.dataset.lower() == "biwi":
        if args.fps == 30:
            args.fps = 25
        if args.vertice_dim == 5023 * 3:
            args.vertice_dim = 23370 * 3

    # wav_dir 기본값: <dataset>/wav  (ex: vocaset/wav)
    if args.wav_dir is None:
        args.wav_dir = os.path.join(args.dataset, "wav")

    # 템플릿 메쉬 (FLAME_sample.ply) 읽기
    template_file = os.path.join(
        args.dataset, args.render_template_path, "FLAME_sample.ply"
    )
    print(f"[INFO] Using template: {template_file}")
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")

    template_trimesh = trimesh.load(template_file, process=False)
    faces = np.asarray(template_trimesh.faces).astype(np.int64)

    # 입력 npy 폴더
    input_path = args.pred_path
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"pred_path is not a directory: {input_path}")

    files = sorted(f for f in os.listdir(input_path) if f.endswith(".npy"))

    # 출력 비디오 폴더
    output_path = os.path.join(args.output, "videos")
    os.makedirs(output_path, exist_ok=True)

    print(f"[INFO] Found {len(files)} npy files in {input_path}")
    print(f"[INFO] wav_dir = {args.wav_dir}")
    print(f"[INFO] with_audio = {args.with_audio}")

    for file in files:
        predicted_vertices_path = os.path.join(input_path, file)
        print(f"[RENDER] {predicted_vertices_path}")

        predicted_vertices = np.load(predicted_vertices_path)
        # (frame, vertex_num, 3)
        predicted_vertices = np.reshape(
            predicted_vertices,
            (-1, args.vertice_dim // 3, 3),
        )

        render_sequence_meshes(
            args,
            predicted_vertices,
            faces,
            output_path,
            predicted_vertices_path,
        )


if __name__ == "__main__":
    main()
