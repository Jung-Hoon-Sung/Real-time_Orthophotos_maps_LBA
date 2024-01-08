import time
import cv2
from .exif import get_metadata
from .exif import restore_orientation
from .eo import geographic2plane
from .eo import rpy_to_opk
from .eo import rot_3d
import numpy as np
from rich.console import Console

import subprocess
import os
import shutil
from .colmap.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec
import pandas as pd

import sys
sys.path.append('/source/OpenSfM')
from opensfm import dataset
from opensfm import commands
from argparse import Namespace
from PIL import Image

console = Console()


def query_points(reconstruction, tracks, target_image):
    """
    Query point cloud by track_id of a target image
    recontruction: path of reconstruction.json
    tracks: path of tracks.csv
    """
    df_reconstruction = pd.read_json(reconstruction)    
    points = df_reconstruction["points"][0]
    print(f" * [Before] no. of points: {len(points)}")

    df_tracks = pd.read_csv(tracks, skiprows=1, sep='\t+', header=None)
    df_tracks.columns = ["image", "track_id", "feature_id", "feature_x", "feature_y", "feature_s", 
                         "r", "g", "b", "segmentation", "instance"]

    # extract track_id
    print(f" * {target_image} is selected")
    tracks_id = df_tracks.loc[df_tracks['image'] == target_image]["track_id"]
    
    # query points
    tmp = []
    for key in tracks_id:
        row = points.get(str(key))
        if row is None:
            continue
        color, coordinates = row["color"], row["coordinates"]
        # id, r, g, b, x, y, z
        tmp.append([key, coordinates[0], coordinates[1], coordinates[2], 
                    color[0], color[1], color[2]])
    new_points = np.array(tmp)

    unwanted = set(points) - set(new_points[:, 0].astype(int).astype(str))
    for unwanted_key in unwanted:
        del points[unwanted_key]
    print(f" * [After] no. of points: {len(points)}")

    df_reconstruction.to_json(reconstruction, orient='records')


def eo_from_opensfm_colmap(path):
    # position
    with open(os.path.join(path, "image_geocoords.tsv"), "r") as f:
        eos = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            elems = line.split("\t")
            if len(elems) > 0 and elems[0] != "Image":                
                image_name = elems[0]
                X, Y, Z = map(float, elems[1:])
                eos.append([image_name, X, Y, Z])
    eos.sort()    
    
    # orientation - rotation matrix
    _, images, _ = read_model(os.path.join(path, "colmap_export"), ext=".txt")
    for img in images.values():
        if img.name == eos[-1][0]:
            R = qvec2rotmat(img.qvec)
            R[:, 1] = -R[:, 1]
            R[:, 2] = -R[:, 2]
            R = R.T   
    
    print(eos[-1][1:], R)

    return eos[-1][1:], R


def direct_georeferencing(image_path, sensor_width, epsg):
    print('Georeferencing - ' + image_path)
    image = cv2.imread(image_path, -1)

    # 1. Extract metadata from a image
    focal_length, orientation, eo, maker = get_metadata(image_path)  # unit: m, _, ndarray

    # 2. Restore the image based on orientation information
    restored_image = restore_orientation(image, orientation)

    image_rows = restored_image.shape[0]
    image_cols = restored_image.shape[1]

    pixel_size = sensor_width / image_cols  # unit: mm/px
    pixel_size = pixel_size / 1000  # unit: m/px

    eo = geographic2plane(eo, epsg)
    opk = rpy_to_opk(eo[3:], maker)
    eo[3:] = opk * np.pi / 180   # degree to radian
    R = rot_3d(eo)

    EO, IO = {}, {}
    EO["eo"] = eo
    EO["rotation_matrix"] = R
    IO["pixel_size"] = pixel_size
    IO["focal_length"] = focal_length

    # console.print(
    #     f"EOP: {eo[0]:.2f} | {eo[1]:.2f} | {eo[2]:.2f} | {eo[3]:.2f} | {eo[4]:.2f} | {eo[5]:.2f}\n"
    #     f"Focal Length: {focal_length * 1000:.2f} mm, Maker: {maker}",
    #     style="blink bold red underline")

    return restored_image, EO, IO

def lba_opensfm(images_path, target_image_path, sensor_width=6.16, epsg=5186):
    data = dataset.DataSet(images_path)
    
    # Create a Namespace object for command line arguments
    args = Namespace(
        dataset=images_path, 
        algorithm='incremental', 
        reconstruction='reconstruction.json',
        reconstruction_index=0,
        tracks='tracks.csv',
        # output='geo_coords.json',
        output= 'undistorted',
        # output=os.path.join('undistorted', 'geo_coords.json'),
        skip_images=False,
        subfolder='undistorted',
        interactive=False,

        proj='+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 +ellps=GRS80 +units=m +no_defs',
        transformation=False,  # 좌표 변환 매트릭스를 출력하려면 True로 설정
        image_positions=False,  # 이미지 위치를 내보내려면 True로 설정
        # reconstruction=True,  # reconstruction.json을 내보내려면 True로 설정
        dense=False,  # 밀도가 높은 포인트 클라우드를 내보내려면 True로 설정
        # output=None,  # 출력 파일의 경로 (데이터셋에 상대적). 필요한 경우 설정
        no_cameras=False,  # 카메라 위치를 저장하지 않는 경우를 위한 값
        no_points=False,  # 포인트를 저장하지 않는 경우를 위한 값
        depthmaps=False,  # 이미지별 깊이 맵을 포인트 클라우드로 내보내는 경우를 위한 값
        point_num_views=False,  # 각 포인트와 연관된 관찰 수를 내보내는 경우를 위한 값
    )

    # extract_metadata
    print("Starting extract_metadata...")
    start_time = time.time()
    commands.extract_metadata.Command().run(data, args)
    print(f"Finished extract_metadata in {time.time() - start_time:.2f} seconds.\n")

    # detect_features
    print("Starting detect_features...")
    start_time = time.time()
    commands.detect_features.Command().run(data, args)
    print(f"Finished detect_features in {time.time() - start_time:.2f} seconds.\n")

    # match_features
    print("Starting match_features...")
    start_time = time.time()
    commands.match_features.Command().run(data, args)
    print(f"Finished match_features in {time.time() - start_time:.2f} seconds.\n")

    # create_tracks
    print("Starting create_tracks...")
    start_time = time.time()
    commands.create_tracks.Command().run(data, args)
    print(f"Finished create_tracks in {time.time() - start_time:.2f} seconds.\n")

    # reconstruct
    print("Starting reconstruct...")
    start_time = time.time()
    commands.reconstruct.Command().run(data, args)
    print(f"Finished reconstruct in {time.time() - start_time:.2f} seconds.\n")

    # Query points by track_id in target_image
    query_points(os.path.join(images_path, "reconstruction.json"), 
                os.path.join(images_path, "tracks.csv"), os.path.basename(target_image_path))

    # export_geocoords for Image Positions
    print("Starting export_geocoords for Image Positions...")

    args.output = os.path.join(images_path, 'image_geocoords.tsv')
    args.proj = '+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 +ellps=GRS80 +units=m +no_defs'
    args.image_positions = True 
    args.reconstruction = False

    start_time = time.time()
    commands.export_geocoords.Command().run(data, args)
    print(f"Finished export_geocoords for Image Positions in {time.time() - start_time:.2f} seconds.\n")

    # export_colmap
    print("Starting export_colmap...")
    # I'm assuming you have a 'binary' parameter in your export_colmap.Command() class, 
    # if not, please remove the next line.
    args.binary = False
    commands.export_colmap.Command().run(data, args)
    print("Finished export_colmap.\n")

    # export_geocoords for Reconstruction
    print("Starting export_geocoords for Reconstruction...")
    args.output = os.path.join(images_path, 'reconstruction.geocoords.json')
    args.image_positions = False
    args.reconstruction = True

    start_time = time.time()
    commands.export_geocoords.Command().run(data, args)
    print(f"Finished export_geocoords for Reconstruction in {time.time() - start_time:.2f} seconds.\n")

    # Rename the generated geocoords reconstruction file
    shutil.move(os.path.join(images_path, 'reconstruction.geocoords.json'), 
                os.path.join(images_path, 'reconstruction.json'))


    # export_ply
    print("Starting export_ply...")
    args.no_cameras = True
    start_time = time.time()
    commands.export_ply.Command().run(data, args)
    print(f"Finished export_ply in {time.time() - start_time:.2f} seconds.\n")

    # #TODO: Override exif - exif_overrides.json in data_path
    # # example:
    # # {
    # #     "image_name.jpg": {
    # #         "gps": {
    # #             "latitude": 52.51891,
    # #             "longitude": 13.40029,
    # #             "altitude": 27.0,
    # #             "dop": 5.0
    # #         }
    # #     }
    # # }
    # # should replace data["image_name.jpg"]["gps"]["latitude"] to computed latitude

    focal_length, orientation, _, _ = get_metadata(target_image_path)  # unit: m, _, ndarray
    
    with Image.open(target_image_path) as img:
        image_data = np.array(img)
        image_cols, image_rows = img.size

    pixel_size = sensor_width / image_cols / 1000  # unit: m/px
    
    pos, R = eo_from_opensfm_colmap(images_path)
    print("pos:", pos)
    print("R:", R)
    pos = np.array(pos)
    EO, IO = {}, {}
    EO["eo"] = pos
    EO["rotation_matrix"] = R
    IO["pixel_size"] = pixel_size
    IO["focal_length"] = focal_length
    print("EO:", EO)
    print("IO:", IO)
    
    return image_data, EO, IO

if __name__ == "__main__":
    lba_opensfm(images_path='data/yangpyeong', sensor_width=6.3)
    eo_from_opensfm_colmap("data/yangpyeong")
