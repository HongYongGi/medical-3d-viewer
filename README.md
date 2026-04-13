# Medical 3D Viewer

CT 영상 세그멘테이션 분석 및 3D 시각화 플랫폼. nnUNet v2 기반 자동 세그멘테이션, MPR 다면 재구성, 인터랙티브 3D 렌더링을 지원합니다.

## Features

- **AI 자동 세그멘테이션** - nnUNet v2 통합, 단일 모델 또는 다중 모델 파이프라인 실행
- **MPR 다면 재구성** - Axial / Sagittal / Coronal / Oblique 슬라이스 + 3-Panel 연동 뷰
- **3D 시각화** - Go 기반 Three.js 렌더러 (Plotly 폴백 지원)
- **윈도우 프리셋** - Lung, Bone, Mediastinum 등 원클릭 CT Window/Level
- **세그멘테이션 편집기** - Erosion, Dilation, 라벨 병합/삭제, HU 필터
- **측정 도구** - 거리 및 면적 측정
- **DICOM 지원** - ZIP 업로드 시 자동 NIfTI 변환
- **내보내기** - NIfTI (.nii.gz) 및 STL (3D 프린팅용) 다운로드
- **환자 DB** - SQLite 기반 분석 이력 관리

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Streamlit, Plotly |
| 3D Renderer | Go 1.22, gorilla/mux, gorilla/websocket, Three.js |
| AI Engine | PyTorch, nnUNet v2 |
| Medical I/O | nibabel, pydicom, dicom2nifti, scikit-image |
| Database | SQLite |

## Project Structure

```
medical-3d-viewer/
├── src/medical_viewer/        # Python 메인 패키지
│   ├── app.py                 # Streamlit 앱 진입점
│   ├── core/                  # 설정, DB, 세션, 내보내기
│   ├── inference/             # nnUNet 추론, 파이프라인, 모델 스캐너
│   ├── mpr/                   # MPR 슬라이서, 윈도우, 측정
│   ├── renderer/              # Go 렌더러 HTTP 클라이언트
│   └── ui/                    # Streamlit UI 컴포넌트
├── go_renderer/               # Go 3D 렌더러 서비스
│   ├── cmd/renderer/          # 진입점
│   ├── internal/server/       # HTTP/WebSocket 서버
│   ├── internal/mesh/         # Marching Cubes, 메쉬 심플리파이
│   └── internal/volume/       # NIfTI 볼륨 로더
├── configs/                   # app.yaml, models.yaml
├── tests/                     # 단위 테스트
├── Dockerfile.python          # Python 컨테이너
├── Dockerfile.go              # Go 컨테이너
└── docker-compose.yaml        # 멀티 서비스 오케스트레이션
```

## Quick Start

### Docker (권장)

```bash
docker compose up -d --build
```

- Streamlit UI: http://localhost:8501
- Go Renderer: http://localhost:8080

### 로컬 실행

```bash
# 1. Python 패키지 설치
pip install -e ".[nnunet]"

# 2. Streamlit 실행 (터미널 1)
make run

# 3. Go 렌더러 실행 (터미널 2)
make run-renderer
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make run` | Streamlit 앱 실행 (port 8501) |
| `make run-renderer` | Go 3D 렌더러 실행 (port 8080) |
| `make install` | Python 패키지 설치 (nnUNet 포함) |
| `make dev` | 개발 의존성 설치 (pytest, ruff) |
| `make build-renderer` | Go 렌더러 컴파일 |
| `make clean` | 업로드/결과/메쉬 데이터 삭제 |

## Configuration

### configs/app.yaml

```yaml
app:
  name: "Medical 3D Viewer"
  port: 8501

renderer:
  host: localhost
  port: 8080

paths:
  uploads: data/uploads
  results: data/results
  models: data/models

nnunet:
  weight_dir: /path/to/nnUNet_results
  auto_scan: true
```

### configs/models.yaml

모델과 파이프라인을 정의합니다:

```yaml
models:
  - id: aorta_seg
    name: "TAVR Aorta Segmentation"
    dataset_id: 302
    labels:
      1: "대동맥"
      2: "좌심실"
      # ...

pipelines:
  - id: tavr_full
    name: "TAVR 종합 분석"
    steps:
      - model_id: aorta_seg
        priority: 1
      - model_id: vessel_total
        priority: 2
    merge_strategy: union
```

## Application Pages

| Page | Description |
|------|-------------|
| **Viewer** | CT 업로드, AI 세그멘테이션, MPR/3D 시각화 |
| **Editor** | 세그멘테이션 수동 편집 (모폴로지 연산) |
| **History** | 환자별 분석 이력 조회 |
| **Models** | nnUNet 모델 가중치 관리 |
| **Settings** | 서비스 상태, 데이터 관리 |

## nnUNet 없이 사용

nnUNet이 설치되지 않아도 다음 기능을 사용할 수 있습니다:
- CT NIfTI 파일 MPR 뷰어
- 기존 세그멘테이션 로드 및 시각화
- 세그멘테이션 편집 및 내보내기
- 3D 렌더링

```bash
pip install -e .  # nnUNet 제외 설치
```

## Testing

```bash
make dev
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- Go >= 1.22 (3D 렌더러)
- CUDA (GPU 추론, 선택사항)

## License

MIT
