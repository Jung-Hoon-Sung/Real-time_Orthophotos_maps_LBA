# FROM hwiyoung/opensfm:220912
FROM sungjunghoon/opensfm_hy:latest

# 패키지 업데이트 및 필요한 툴 설치
RUN apt-get update && apt-get install -y \
    gdal-bin \
    pdal \
    libpdal-dev \
    python3-pdal

# Python 패키지 설치
RUN pip install numpy --upgrade && \
    pip install pyexiv2 numba rich pandas scipy pdal && \
    pip install \
    llvmlite==0.39.1 \
    markdown-it-py==2.2.0 \
    mdurl==0.1.2 \
    numba==0.56.4 \
    numpy==1.23.5 \
    opencv-python==4.7.0.72 \
    pyexiv2==2.8.1 \
    Pygments==2.15.1 \
    rich==13.3.4 \
    setuptools==57.4.0 \
    wheel==0.40.0 \
    pytest-shutil==1.7.0 \
    open3d

# 로컬 디렉토리의 모든 내용을 /code로 복사
COPY . /code

# 작업 디렉토리 설정
WORKDIR /code
