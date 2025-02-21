# colmap-custom

`colmap-custom`은 오리지널 [COLMAP](https://github.com/colmap/colmap)를 포크하여 CLI 중심으로 활용하기 위해 제작된 커스텀 버전입니다.  
이미지 기반의 Structure-from-Motion(SfM) 및 3D 재구성 파이프라인을 제공합니다.

> **참고**: COLMAP에 대한 전반적인 설명과 GUI 사용법 등은 [공식 문서](https://colmap.github.io/)를 참조하세요.

---

## 주요 특징

- **원본 COLMAP**의 핵심 기능(Feature Extraction/Matching, Sparse Reconstruction, Dense Reconstruction 등)을 포함  
- **CLI 중심**으로 작동하도록 포크 버전 일부 스크립트 및 설정 수정  
- **CUDA 가속**(선택적) 지원

---

## 1. 빌드 방법

### 1.1. 사전 요구사항

- C++ 컴파일러 (C++11 이상)  
- CMake (버전 3.5 이상 권장)  
- (선택) CUDA (GPU 가속 사용 시 권장)  
- OpenGL, Glew, GLFW 등 그래픽 관련 라이브러리 (COLMAP에서 시각화 기능 사용 시 필요)  

기본 환경(우분투 예시):

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake \
    libatlas-base-dev libsuitesparse-dev \
    libeigen3-dev libgoogle-glog-dev libgflags-dev \
    libqt5core5a libqt5gui5 libqt5opengl5 libqt5widgets5 \
    qtbase5-dev \
    libglew-dev glew-utils \
    freeglut3-dev \
    libcgal-dev
# GPU 사용 시:
# sudo apt-get install nvidia-cuda-toolkit
```

### 1.2. 빌드 절차

```bash
# 1) 소스 다운로드
git clone https://github.com/사용자이름/colmap-custom.git
cd colmap-custom

# 2) 빌드 폴더 생성
mkdir build && cd build

# 3) CMake 설정
cmake ..

# 4) 컴파일
make -j8  # CPU 코어 수에 맞게 -j 옵션 조정
```

> 빌드 완료 후, `build/src/exe` (또는 `build/src/exe/Release`) 폴더 등에 `colmap` 실행 파일이 생성됩니다.

---

## 2. CLI 모드 사용법

### 2.1. 실행 파일 위치에서 직접 실행

환경변수를 별도 설정하지 않았다면, 다음과 같이 **절대경로 혹은 상대경로**로 직접 실행할 수 있습니다:
```bash
# 빌드 폴더 내에서:
cd colmap-custom/build/src/exe
./colmap help
```
또는:
```bash
/home/username/colmap-custom/build/src/exe/colmap gui
```
등으로 실행 가능합니다.

### 2.2. PATH 설정 혹은 alias (선택)

자주 사용한다면 `.bashrc`(또는 `.zshrc`) 파일에 경로를 추가해 편리하게 쓸 수 있습니다:

```bash
export PATH="$HOME/colmap-custom/build/src/exe:$PATH"
# 반영
source ~/.bashrc
```

이제 어느 디렉토리에서든지 `colmap` 명령어로 바로 실행할 수 있습니다.

---

## 3. 예시 워크플로우 (CLI 중심)

일반적인 SfM ~ MVS 재구성 파이프라인은 다음 단계를 거칩니다.

> **전제**:  
> - 이미지들이 `path/to/images` 폴더에 있음  
> - 새로운 데이터베이스 파일을 `path/to/database.db`로 생성  
> - 결과물을 저장할 `path/to/sparse`, `path/to/dense` 폴더 준비  

1. **특징점 추출 (Feature Extraction)**

   ```bash
   colmap feature_extractor \
       --database_path path/to/database.db \
       --image_path path/to/images \
       --SiftExtraction.use_gpu 1
   ```
   - `SiftExtraction.use_gpu=1`은 CUDA 설치 후 GPU를 활용

2. **특징점 매칭 (Feature Matching)**

   ```bash
   colmap exhaustive_matcher \
       --database_path path/to/database.db \
       --SiftMatching.use_gpu 1
   ```
   - 모든 이미지 쌍에 대해 특징점을 매칭 (이미지 수가 많으면 시간이 오래 걸림)

3. **Mapper (Sparse Reconstruction)**

   ```bash
   mkdir path/to/sparse
   colmap mapper \
       --database_path path/to/database.db \
       --image_path path/to/images \
       --output_path path/to/sparse
   ```
   - Sparse reconstruction 결과(카메라 포즈, 3D 포인트 클라우드) 생성

4. **Dense Reconstruction** (옵션)

   1) **왜곡보정 (Image Undistorter)**
      ```bash
      mkdir path/to/dense
      colmap image_undistorter \
          --image_path path/to/images \
          --input_path path/to/sparse/0 \
          --output_path path/to/dense \
          --output_type COLMAP
      ```

   2) **PatchMatchStereo & StereoFusion**
      ```bash
      colmap patch_match_stereo \
          --workspace_path path/to/dense \
          --workspace_format COLMAP \
          --PatchMatchStereo.gpu_index 0

      colmap stereo_fusion \
          --workspace_path path/to/dense \
          --workspace_format COLMAP \
          --input_type geometric \
          --output_path path/to/dense/fused.ply
      ```
      - `fused.ply`는 포인트 클라우드 형태

5. **메시 생성 (선택)**

   ```bash
   colmap poisson_mesher \
       --input_path path/to/dense/fused.ply \
       --output_path path/to/dense/meshed-poisson.ply
   ```
   - Poisson Reconstruction 알고리즘을 이용해 3D 메시 생성

---

## 4. FAQ

1. **CUDA 관련 에러**  
   - CUDA가 제대로 깔려있지 않거나 버전이 맞지 않으면 GPU 기능이 인식되지 않을 수 있습니다.  
   - `--SiftExtraction.use_gpu 0` 또는 `--SiftMatching.use_gpu 0`으로 CPU 모드로 실행하면 우회 가능하지만, 속도가 느려집니다.

2. **`colmap` 명령어를 못 찾음**  
   - 빌드 폴더에서 직접 `./colmap` 형태로 실행하거나, 위에 나온 **PATH**/**alias** 설정이 되어 있는지 재확인하세요.

3. **`GL/glew.h: No such file or directory` 등 빌드 에러**  
   - OpenGL 관련 라이브러리가 설치되지 않았을 가능성이 큽니다. OS에 맞춰 `libglew-dev`(우분투) 등 의존 패키지를 설치해야 합니다.

---

## 5. 라이선스

- 본 레포지토리는 오리지널 [COLMAP](https://github.com/colmap/colmap)의 소스코드를 기반으로 합니다.  
- COLMAP은 [BSD 3-Clause License](https://github.com/colmap/colmap/blob/dev/LICENSE.txt)를 따릅니다.  
- 이 포크 버전(`colmap-custom`)도 동일한 라이선스 하에 배포됩니다.

---

## 6. 기여 방법

1. 이 레포지토리를 포크하고, 새로운 브랜치에서 개발하세요.  
2. 변경 사항을 커밋하고 PR(Pull Request)을 보내주세요.  
3. Issue Tracker를 통해 버그나 제안 사항을 공유해주시면 큰 도움이 됩니다.

---

### 문의 / 컨택

- Repository Issues 활용  
