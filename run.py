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
# profile_times = []
write_times = []

def shift_data_to_end(data_list, max_len):
    """Move the data to the end of the list, padding the start with zeros."""
    return [0] * (max_len - len(data_list)) + data_list

def display_processing_times():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Process")

    # Find the max length of the data lists
    max_len = max(len(georef_times), len(dem_times), len(rectify_times), len(write_times), len(copy_image_times))

    # Create the Orthophoto columns based on max_len
    for i in range(max_len):
        table.add_column(f"Orthophoto {i+1} (s)", justify="right")
    table.add_column("Total (s)", justify="right")

    headers = ["Georeferencing", "DEM", "Rectify", "Write", "Copy_Images"]
    data_lists = [georef_times, dem_times, rectify_times, write_times, copy_image_times]

    # Shift all data to the right based on max_len
    data_lists = [shift_data_to_end(data, max_len) for data in data_lists]
    
    # Calculate sums for each orthophoto and total
    sum_per_image = [sum(times) for times in zip(*data_lists)]
    sum_total = sum(sum_per_image)

    # Add rows to the table
    for header, data_list in zip(headers, data_lists):
        row_data = [header] + [f"{value:.6f}" if value != 0 else " " for value in data_list] + [f"{sum(data_list):.6f}"]
        table.add_row(*row_data)

    # Add the total row
    total_row_data = ["Total"] + [f"{value:.6f}" for value in sum_per_image] + [f"{sum_total:.6f}"]
    table.add_row(*total_row_data)

    console.print(table)

def save_processing_times_to_csv():
    with open('/code/output/processing_times.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Find the max length of the data lists
        max_len = max(len(georef_times), len(dem_times), len(rectify_times), len(write_times), len(copy_image_times))
        
        # Shift all data to the right based on max_len
        data_lists = [shift_data_to_end(data, max_len) for data in [georef_times, dem_times, rectify_times, write_times, copy_image_times]]

        # Headers for the CSV
        headers = ["Process"] + [f"Orthophoto_{i+1}_(s)" for i in range(max_len)] + ["Total_(s)"]
        csv_writer.writerow(headers)

        process_names = ["Georeferencing", "DEM", "Rectify", "Write", "Copy_Images"]
        for process, data_list in zip(process_names, data_lists):
            row_data = [process] + [f"{value:.6f}" if value != 0 else "" for value in data_list] + [f"{sum(data_list):.6f}"]
            csv_writer.writerow(row_data)

        # Add the total row
        sum_per_image = [sum(times) for times in zip(*data_lists)]
        total_row_data = ["Total"] + [f"{value:.6f}" for value in sum_per_image] + [f"{sum(sum_per_image):.6f}"]
        csv_writer.writerow(total_row_data)
        
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
    georef_times.append(georef_time)
    console.print(f"Georeferencing time: {georef_time:.2f} sec", style="blink bold red underline")

    # 2. Generate DEM
    dem_start = time.perf_counter()
    bbox = boundary(image, IO, EO, args.ground_height)
    dem_time = time.perf_counter() - dem_start
    dem_times.append(dem_time)
    console.print(f"DEM time: {dem_time:.2f} sec", style="blink bold red underline")

    # 3. Rectify
    rectify_start = time.perf_counter()
    pixel_size, focal_length = IO["pixel_size"], IO["focal_length"]
    pos, rotation_matrix = EO["eo"], EO["rotation_matrix"]
    b, g, r, a = rectify_plane_parallel(image, pixel_size, focal_length, pos, rotation_matrix, 
                                        args.ground_height, bbox, args.gsd)
    bbox = bbox.ravel()
    rectify_time = time.perf_counter() - rectify_start
    rectify_times.append(rectify_time)
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
    
    start_time = time.perf_counter()
    
    image_list = sorted(os.listdir(src_folder))
    if image_list:
        last_image = image_list[-1]
        src_path = os.path.join(src_folder, last_image)
        dest_path = os.path.join(dest_folder, last_image)
        shutil.copy(src_path, dest_path)
        
        copy_time = time.perf_counter() - start_time
        copy_image_times.append(copy_time)
        return dest_path
    else:
        return None
    
def copy_images_to_destination(src_folder, dest_folder, start_idx, end_idx):
    start_time = time.perf_counter()

    # 선택된 이미지들을 대상 폴더로 복사, 이미 존재하는 이미지는 다시 복사하지 않음
    image_list = sorted(os.listdir(src_folder)) # src_folder : total_images
    copied_images = image_list[start_idx:end_idx] # start_idx 
    for image in copied_images:
        if image not in os.listdir(dest_folder):
            shutil.copy(os.path.join(src_folder, image), os.path.join(dest_folder, image))

    dest_images = sorted(os.listdir(dest_folder))
    while len(dest_images) > 5:
        oldest_image = dest_images.pop(0)  # 이미 정렬된 리스트에서 첫 번째 요소 확인
        os.remove(os.path.join(dest_folder, oldest_image))  # 가장 오래된 이미지를 제거

    copy_time = time.perf_counter() - start_time
    copy_image_times.append(copy_time)
    return copied_images

def orthophoto_process_image(args, image_path, is_early=False):
    if is_early:
        b, g, r, a, bbox = orthophoto_direct(args, image_path)
    else:
        try:
            r, g, b, a, bbox = orthophoto_lba(args, image_path)
        except Exception:
            b, g, r, a, bbox = orthophoto_direct(args, image_path)

    dst_filename = os.path.splitext(os.path.basename(image_path))[0]
    dst_path = os.path.join(args.output_path, dst_filename)
    
    write_start = time.perf_counter()
    create_pnga_optical(b, g, r, a, bbox, args.gsd, args.epsg_out, dst_path)
    write_time = time.perf_counter() - write_start
    write_times.append(write_time)
    console.print(f"Write time: {write_time:.2f} sec", style="blink bold red underline")
    
    return image_path

def main():
    parser = argparse.ArgumentParser(description="Run Orthophoto_Maps")
    parser.add_argument("--input_project", help="path to the input project folder", 
                        default="/source/OpenSfM/data/test_images/")
    parser.add_argument("--metadata_in_image", help="images have metadata?", default=True)    
    parser.add_argument("--output_path", help="path to output folder", default="output/")
    parser.add_argument("--no_image_process", type=int, help="the number of images to process at once", 
                        default=5)
    parser.add_argument("--sys_cal", choices=["DJI", "samsung"],
                        help="types of a system calibration", default="DJI")
    parser.add_argument("--epsg_out", type=int, help="EPSG of output data", default=5186)
    parser.add_argument("--gsd", type=float,help="target ground sampling distance in m. set to 0 to disable", 
                        default=0.1)
    parser.add_argument("--dem", choices=["dsm", "dtm", "plane"],
                        help="types of projection plane", default="plane")
    parser.add_argument("--ground_height", type=int,
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

    if len(all_images) >= args.no_image_process:
        for i in range(0, len(all_images) - args.no_image_process + 1):
            copied_images = copy_images_to_destination(total_images_folder, images_folder, i, i + args.no_image_process)

            # 초반 이미지에 대해 direct 처리
            if i == 0:
                for idx in range(min(4, len(copied_images))):
                    early_image_path = os.path.join(images_folder, copied_images[idx])
                    orthophoto_process_image(args, early_image_path, is_early=True)

            # 마지막 이미지 처리
            last_image_path = os.path.join(images_folder, copied_images[-1])
            orthophoto_process_image(args, last_image_path)

    else:
        for img in all_images:
            copied_image_path = os.path.join(images_folder, img)
            if not os.path.exists(copied_image_path):
                shutil.copy(os.path.join(total_images_folder, img), copied_image_path)
            
            if copied_image_path:
                orthophoto_process_image(args, copied_image_path, is_early=True)

if __name__ == '__main__':
    
    start_time = datetime.now()

    start_message = Text("Starting the process...", style="bold yellow")
    start_icon = Text("[⏳]", style="bold yellow")
    combined_start_message = start_icon + " " + start_message
    console.print(Panel(combined_start_message, border_style="yellow", padding=(1, 2)))

    main()
    
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
