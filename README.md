# COLMAP Pipeline with Python Scripts

본 리포지토리는 **이미 빌드된 COLMAP** 실행 바이너리와 **Python 스크립트**(task.py, exe.py 등)를 통해 간단히 3D 복원을 진행할 수 있도록 하는 방법을 안내합니다.

이미지는 Linux64 서버 환경에서 빌드되어있습니다. 만약 다른 환경을 사용하고 계시다면
Colmap 원본 레포지토리를 활용해 빌드하셔야 합니다.

## 폴더 구조

```
.
├── application/
│   ├── build/              # 이미 빌드된 COLMAP 바이너리 경로
│   └── exe.py              # COLMAP 관련 함수를 모아 둔 Python 코드 (mapper, dense_reconstruction 등)
├── inputs/
│   └── example_dataset/    # 원본 이미지 폴더
├── outputs/
│   └── example_output/     # 재구성 결과가 저장될 폴더
├── task.py                 # 파이프라인을 제어하는 Python 스크립트
└── README.md               # 사용 안내 (이 문서)
```

> **주의**:  
> - `application/` 폴더에는 colmap 바이너리가 이미 빌드되어 있음.  
> - `exe.py` 내 `COLMAP_EXE` 변수가 이 바이너리를 가리키도록 설정되어 있어야 합니다. (예: `COLMAP_EXE = os.path.abspath("./colmap")` 혹은 절대 경로.)

---

## 1. 의존성 / 환경

- **Python 3** (3.6 이상 권장)  
- COLMAP이 빌드되어 있는 **application/build** 폴더  
- 기타 python 라이브러리(표준 라이브러리: `os`, `subprocess`, `shutil`, `logging` 등)

GPU를 사용하려면 **CUDA 호환 드라이버**가 설치된 상태여야 합니다.

---

## 2. 사용 방법

### 2.1. 파이프라인 한 번에 실행

1. **이미지 폴더 준비**  
   - 예: `inputs/desk-images` 폴더에 재구성할 이미지들(예: `IMG_XXXX.jpg`)을 모두 넣어둡니다.

2. **task.py 수정**  
   - `task.py`의 상단에서 `input_path`, `output_path` 변수를 설정:
     ```python
     input_path = "inputs/desk-images"     # 원본 이미지 폴더
     output_path = "outputs/3dgs-images"   # 결과 저장 경로
     GPU_INDEX = 0                         # 0: 첫 번째 GPU, -1: CPU, 1: 두 번째 GPU 등
     ```
   - GPU 사용을 원하면 `GPU_INDEX = 0` (혹은 원하는 GPU 번호), CPU만 쓰려면 `-1`로 지정.

3. **`all_pipeline()` 함수 호출**  
   - `task.py` 내부에서 다음과 같이 `all_pipeline(input_path, output_path, gpu_index=GPU_INDEX)` 함수를 호출하면, **한 번에 전체 파이프라인**을 진행합니다:
     ```python
     all_pipeline(input_path, output_path, GPU_INDEX)
     ```
     1. **prepare**: 이미지를 `outputs/.../images` 폴더로 복사(또는 symlink)  
     2. **feature_extraction**: SIFT 특징점 추출  
     3. **match_features**: 특징점 매칭  
     4. **mapper**: Sparse Reconstruction  
     5. **dense_reconstruction**: Dense Reconstruction

4. **task.py 실행**  
   ```bash
   python3 task.py
   ```
   - 완료 후 `outputs/3dgs-images/sparse/` 폴더에서 스파스 포인트 클라우드,  
   - `outputs/3dgs-images/dense/` 폴더에서 뎁스맵(`fused.ply`) 등을 확인할 수 있습니다.  

이와 같이 **`all_pipeline()`** 함수를 사용하면 COLMAP 파이프라인 전체를 자동으로 돌릴 수 있으므로, 별도의 단계별 호출 없이도 쉽게 3D 복원을 진행할 수 있습니다.  

### 2.2. 단계별로 실행

원한다면 `all_pipeline` 대신 **단계별 함수**를 수동으로 호출할 수도 있습니다. 예:

```python
prepare(input_path, output_path)
feature_extraction(output_path, gpu_index=0)  # SIFT 추출
match_features(output_path, gpu_index=0)      # 특징점 매칭
mapper(output_path)                           # Sparse
dense_reconstruction(output_path, gpu_index=0)# Dense
```

`gpu_index` 인자를 통해 원하는 GPU를 지정하거나 `-1`로 CPU 모드로 실행 가능합니다.

---

## 3. GPU 인덱스 설정 & CPU 모드

- `gpu_index >= 0` → 해당 번호 GPU 사용.  
- `gpu_index = -1` → CPU 모드.  
- COLMAP의 **PatchMatchStereo** 단계에서는 명시적으로 `--PatchMatchStereo.gpu_index`를 사용합니다.  
- **SIFT 추출** / **특징점 매칭** 부분은 본 코드에서는 `CUDA_VISIBLE_DEVICES` 환경변수를 통해 강제로 해당 GPU만 보이게 하므로, `gpu_index`가 1이라면 `CUDA_VISIBLE_DEVICES="1"`로 설정해주게 됩니다.

---

## 4. 자주 묻는 질문(FAQ)

1. **이미 `IMAGE_EXISTS` 로그가 뜹니다.**  
   - 이는 기존에 같은 이미지 이름으로 이미 특징점이 추출된 기록이 `database.db`에 존재함을 의미합니다.  
   - 새로 추출을 원하면 `outputs/.../database.db`를 삭제한 후 다시 시도하세요.

2. **GPU 메모리 부족 / CUDA 에러**  
   - `nvidia-smi`로 GPU 메모리 사용을 확인하십시오.  
   - `gpu_index=-1`로 CPU 모드로 동작시키거나, `--max_image_size`나 스레드 수를 제한하여 메모리 사용량을 줄일 수 있습니다.

3. **PermissionError**  
   - 보통 `colmap` 심볼릭 링크 또는 실행 파일 권한이 잘못된 경우입니다.  
   - `ls -l application/build/src/exe/colmap` 등을 통해 실행 권한이 있는지, 심볼릭 링크가 올바른지 확인하세요.

4. **'Failed to process the image'** 메시지**  
   - 손상된 JPEG 파일이거나, GPU 연산 실패(CUDA error) 등이 원인일 수 있습니다.  
   - `gpu_index`를 바꿔 다른 GPU로 시도하거나, CPU 모드로 확인해보세요.

---

## 5. 기타

- **추가 기능**:  
  - 메시 생성(예: `poisson_mesher`) 함수를 추가하여 `fused.ply`에서 메시를 만들 수 있습니다.  
  - 큰 규모의 이미지에 대해서는 “hierarchical_mapper” 등 고급 기능도 고려해 볼 수 있습니다.
- **환경 변수**:  
  - 코드 내부에서 `subprocess.run` 시 `env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)` 등을 사용하여 GPU를 동적으로 선택합니다.

---

## 6. 라이선스

- **COLMAP**은 [BSD 3-Clause License](https://github.com/colmap/colmap/blob/dev/LICENSE.txt)를 준수합니다.  
- 본 프로젝트의 Python 스크립트 또한 동일 라이선스로 배포하거나, 내부 정책에 맞춰 라이선스를 명시해 주세요.

---

## 7. 문의 / 기여

- 궁금한 점이나 버그 제보는 Issue 트래커 또는 PR로 보내 주세요.  
- 직접 코드를 수정한 뒤 Pull Request를 통해 기여하실 수 있습니다.  