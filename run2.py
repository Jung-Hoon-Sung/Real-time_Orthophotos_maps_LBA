from module.georeferencing import direct_georeferencing
from module.georeferencing import lba_opensfm
from module.dem import boundary
from module.dem import generate_dem_pdal
from module.rectification import rectify_plane_parallel
from module.rectification import rectify_dem_parallel
from module.rectification import create_pnga_optical
import sys

import argparse
import os
import time
import numpy as np
import csv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import shutil 
import cProfile
import pstats
console = Console()
sensor_width = 6.3  # unit: mm, Mavic

# 초기화: 각 작업의 시간을 기록할 리스트들
georef_times = []
dem_times = []
rectify_times = []
copy_image_times = []
profile_times = []
write_times = []

def display_processing_times():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Process")
    for i in range(len(georef_times)):
        table.add_column(f"Orthophoto {i+1} (s)", justify="right")
    table.add_column("Total (s)", justify="right")  # 추가된 총 합계 열

    headers = ["Georeferencing", "DEM", "Rectify", "Write", "Copy_Images", "Profile"]
    data_lists = [georef_times, dem_times, rectify_times, write_times, copy_image_times, profile_times]

    # 각 'Orthophoto'의 총 합계 계산
    sum_per_image = [sum(times) for times in zip(*data_lists)]
    sum_total = sum(sum_per_image)  # 모든 'Orthophoto'의 합계

    for header, data_list in zip(headers, data_lists):
        table.add_row(header, *[f"{value:.6f}" for value in data_list], f"{sum(data_list):.6f}")  # 각 데이터의 총 합계 추가

    table.add_row("Total", *[f"{value:.6f}" for value in sum_per_image], f"{sum_total:.6f}")  # 총 합계 값 추가

    console.print(table)

def save_processing_times_to_csv():
    with open('/code/output/processing_times.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Define headers and related data lists
        headers = ["Georeferencing", "DEM", "Rectify", "Write", "Copy_Images", "Profile", "Total"]
        data_lists = [georef_times, dem_times, rectify_times, write_times, copy_image_times, profile_times]
        
        # Write headers
        csv_writer.writerow(headers)
        
        # Calculate and write rows for each process time
        total_times = [sum(times) for times in zip(*data_lists)]
        for row in zip(*data_lists, total_times):
            csv_writer.writerow([f"{value:.6f}" for value in row])
        
        # Calculate average times
        avg_values = [sum(times) / len(times) if times else 0 for times in data_lists]
        avg_total = sum(avg_values)
        
        # Write average times
        csv_writer.writerow(["-----"] * 6 + ["-----"])
        csv_writer.writerow([f"{value:.6f}" for value in avg_values] + [f"{avg_total:.6f}"])
        
        # Write total times
        sum_total = sum(total_times)
        csv_writer.writerow(["-----"] * 6 + [f"{sum_total:.6f}"])
        
def cleanup_folder_except(path, exceptions):
    """
    Deletes everything inside the path except for the items in exceptions.
    """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if item not in exceptions:
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)
                
def orthophoto_direct(args, image_path):
    # 1. Georeferencing
    georef_start = time.perf_counter()
    image, EO, IO = direct_georeferencing(image_path=image_path, 
                                            sensor_width=sensor_width,
                                            epsg=args.epsg_out)
    georef_time = time.perf_counter() - georef_start

    console.print(f"Georeferencing time: {georef_time:.2f} sec", style="blink bold red underline")

    # 2. Generate DEM
    dem_start = time.perf_counter()
    bbox = boundary(image, IO, EO, args.ground_height)
    dem_time = time.perf_counter() - dem_start

    console.print(f"DEM time: {dem_time:.2f} sec", style="blink bold red underline")

    # 3. Rectify
    rectify_start = time.perf_counter()
    pixel_size, focal_length = IO["pixel_size"], IO["focal_length"]
    pos, rotation_matrix = EO["eo"], EO["rotation_matrix"]
    b, g, r, a = rectify_plane_parallel(image, pixel_size, focal_length, pos, rotation_matrix, 
                                        args.ground_height, bbox, args.gsd)
    bbox = bbox.ravel()
    rectify_time = time.perf_counter() - rectify_start

    console.print(f"Rectify time: {rectify_time:.2f} sec", style="blink bold red underline")

    return b, g, r, a, bbox


def orthophoto_lba(args, image_path):
    # 1. Georeferencing
    georef_start = time.perf_counter()
    image, EO, IO = lba_opensfm(images_path=args.input_project, target_image_path=image_path,
                                sensor_width=sensor_width, epsg=args.epsg_out)
    georef_time = time.perf_counter() - georef_start
    georef_times.append(georef_time)
    console.print(f"Georeferencing time: {georef_time:.2f} sec", style="blink bold red underline")

    # 2. Generate DEM
    dem_start = time.perf_counter()
    dem_x, dem_y, dem_z, bbox = generate_dem_pdal(os.path.join(args.input_project, "reconstruction.ply"), args.dem, args.gsd)
    dem_time = time.perf_counter() - dem_start
    dem_times.append(dem_time)
    console.print(f"DEM time: {dem_time:.2f} sec", style="blink bold red underline")

    # 3. Rectify
    rectify_start = time.perf_counter()
    pixel_size, focal_length = IO["pixel_size"], IO["focal_length"]
    pos, rotation_matrix = EO["eo"], EO["rotation_matrix"]
    b, g, r, a = rectify_dem_parallel(image, pixel_size, focal_length, pos, rotation_matrix, 
                                        dem_x, dem_y, dem_z)
    rectify_time = time.perf_counter() - rectify_start
    rectify_times.append(rectify_time)
    console.print(f"Rectify time: {rectify_time:.2f} sec", style="blink bold red underline")

    return b, g, r, a, bbox

def copy_last_image_to_destination(src_folder, dest_folder):
    """
    Copies the last image from the source folder to the destination folder.
    Returns the path of the copied image in the destination folder.
    """
    image_list = sorted(os.listdir(src_folder))
    if image_list:
        last_image = image_list[-1]
        src_path = os.path.join(src_folder, last_image)
        dest_path = os.path.join(dest_folder, last_image)
        shutil.copy(src_path, dest_path)
        return dest_path
    else:
        return None
    
def copy_images_to_destination(src_folder, dest_folder, start_idx, end_idx):
    start_time = time.perf_counter()

    # 선택된 이미지들을 대상 폴더로 복사, 이미 존재하는 이미지는 다시 복사하지 않음
    image_list = sorted(os.listdir(src_folder))
    copied_images = image_list[start_idx:end_idx]
    for image in copied_images:
        if image not in os.listdir(dest_folder):
            shutil.copy(os.path.join(src_folder, image), os.path.join(dest_folder, image))

    # images 폴더의 이미지 수가 5장을 초과하는지 확인
    while len(os.listdir(dest_folder)) > 5:
        oldest_image = sorted(os.listdir(dest_folder))[0]  # 가장 오래된 이미지를 가져옴
        os.remove(os.path.join(dest_folder, oldest_image))  # 가장 오래된 이미지를 제거

    copy_time = time.perf_counter() - start_time
    copy_image_times.append(copy_time)
    return copied_images


def main():
    parser = argparse.ArgumentParser(description="Run Orthophoto_Maps")
    parser.add_argument("--input_project", help="path to the input project folder", 
                        default="/source/OpenSfM/data/test_images/")
    parser.add_argument("--metadata_in_image", help="images have metadata?", default=True)    
    parser.add_argument("--output_path", help="path to output folder", default="output/")
    parser.add_argument("--no_image_process", help="the number of images to process at once", 
                        default=5)
    parser.add_argument("--sys_cal", choices=["DJI", "samsung"],
                        help="types of a system calibration", default="DJI")
    parser.add_argument("--epsg_out", help="EPSG of output data", default=5186)
    parser.add_argument("--gsd", help="target ground sampling distance in m. set to 0 to disable", 
                        default=0.1)
    parser.add_argument("--dem", choices=["dsm", "dtm", "plane"],
                        help="types of projection plane", default="plane")
    parser.add_argument("--ground_height", 
                        help="target ground height in m", default=0)
    args = parser.parse_args()

    total_images_folder = os.path.join(args.input_project, "total_images")
    images_folder = os.path.join(args.input_project, "images")

    project_root = os.path.dirname(args.input_project)
    cleanup_folder_except(project_root, ["config.yaml", "total_images"])

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    all_images = sorted(os.listdir(total_images_folder))

    output_folder = os.path.join(args.output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 이미지가 5장 이상인 경우
    if len(all_images) >= 5:
        for i in range(0, len(all_images) - args.no_image_process + 1):
            copied_images = copy_images_to_destination(total_images_folder, images_folder, i, i + args.no_image_process)

            # 마지막 이미지만 처리
            last_image = copied_images[-1]
            image_path = os.path.join(images_folder, last_image)
            try:
                r, g, b, a, bbox = orthophoto_lba(args, image_path)
                
                profile_start = time.perf_counter()
                profile_folder = os.path.join(args.output_path, 'profile')
                if not os.path.exists(profile_folder):
                    os.makedirs(profile_folder)
                src_path = os.path.join(args.input_project, 'profile.log')
                dest_path = os.path.join(profile_folder, f'profile{i + 1}.log')
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                profile_time = time.perf_counter() - profile_start
                profile_times.append(profile_time)

            except Exception as e:
                b, g, r, a, bbox = orthophoto_direct(args, image_path)

            write_start = time.perf_counter()
            temp_output_folder = os.path.join(args.output_path, f'temp_{i + 1}')
            if not os.path.exists(temp_output_folder):
                os.makedirs(temp_output_folder)
            dst_path_for_ortho = os.path.join(temp_output_folder, os.path.splitext(last_image)[0])

            create_pnga_optical(b, g, r, a, bbox, args.gsd, args.epsg_out, dst_path_for_ortho)
        
            write_time = time.perf_counter() - write_start
            write_times.append(write_time)
            console.print(f"Write time: {write_time:.2f} sec", style="blink bold red underline")
            
            # 마지막 루프가 아닐 때만 이미지를 삭제
            if i != (len(all_images) - args.no_image_process):
                os.remove(image_path)

    # 이미지가 5장 미만인 경우
    else:
        copied_last_image_path = copy_last_image_to_destination(total_images_folder, images_folder)
        if copied_last_image_path:
            b, g, r, a, bbox = orthophoto_direct(args, copied_last_image_path)

            write_start = time.perf_counter()
            temp_output_folder = os.path.join(args.output_path, f'temp_1')
            if not os.path.exists(temp_output_folder):
                os.makedirs(temp_output_folder)
            dst_path_for_ortho = os.path.join(temp_output_folder, os.path.splitext(os.path.basename(copied_last_image_path))[0])

            create_pnga_optical(b, g, r, a, bbox, args.gsd, args.epsg_out, dst_path_for_ortho)
        
            write_time = time.perf_counter() - write_start
            write_times.append(write_time)
            console.print(f"Write time: {write_time:.2f} sec", style="blink bold red underline")

            # 이미지 삭제
            os.remove(copied_last_image_path)
            
            
if __name__ == '__main__':
    
    start_time = datetime.now()

    start_message = Text("Starting the process...", style="bold yellow")
    start_icon = Text("[⏳]", style="bold yellow")
    combined_start_message = start_icon + " " + start_message
    console.print(Panel(combined_start_message, border_style="yellow", padding=(1, 2)))

    main()

    # end_time = datetime.now()

    save_processing_times_to_csv()
    display_processing_times()
    
    end_time = datetime.now()
    elapsed_time_seconds = (end_time - start_time).total_seconds()
    
    complete_message = Text("Process has been", style="bold green")
    complete_icon = Text("[✔]", style="bold green")
    complete_message_2 = Text("successfully completed!", style="bold green")
    complete_message_3 = Text(f" / [Elapsed time: {elapsed_time_seconds:.2f} seconds]", style="italic green")

    combined_complete_message = complete_icon + " " + complete_message + " " + complete_message_2 + complete_message_3
    console.print(Panel(combined_complete_message, border_style="green", padding=(1, 2)))
