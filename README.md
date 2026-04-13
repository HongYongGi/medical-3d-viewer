# Medical 3D Viewer

CT NIfTI 영상에 대한 **AI 기반 자동 세그멘테이션**, **MPR 다면 재구성**, **3D 시각화**를 제공하는 의료 영상 분석 도구입니다.

## 주요 기능

- **CT 세그멘테이션** — nnUNet 기반 AI 모델을 통한 자동 장기/혈관 세그멘테이션
- **MPR 뷰어** — Axial, Coronal, Sagittal 및 Oblique 다면 재구성, 윈도잉, 거리/각도 측정
- **3D 시각화** — Go 렌더러(Marching Cubes 메쉬) 또는 Plotly 기반 Isosurface 렌더링
- **세그멘테이션 편집기** — 레이블 브러시/지우개를 통한 수동 보정
- **파이프라인** — 여러 모델을 순차 실행하고 결과를 병합하는 분석 파이프라인
- **내보내기** — NIfTI 세그멘테이션 및 STL 메쉬(3D 프린팅용) 다운로드
- **분석 이력** — 환자 정보 및 분석 결과 이력 관리 (SQLite)
- **Rust 추론 엔진** — ONNX Runtime 기반 고속 추론 (선택)

## 아키텍처

```
┌─────────────────────────────────────────┐
│  Streamlit (Python)          :8501      │
│  ├── UI / MPR 뷰어 / 세그 편집기        │
│  ├── nnUNet 추론 (PyTorch)              │
│  └── Renderer Client (httpx)            │
│            │                            │
│            ▼                            │
│  Go 3D Renderer              :8080      │
│  ├── Marching Cubes 메쉬 생성           │
│  ├── NIfTI 볼륨 로더                    │
│  └── WebSocket 3D 뷰어                 │
│                                         │
│  Rust Inference (선택)                   │
│  └── ONNX Runtime 기반 고속 추론         │
└─────────────────────────────────────────┘
```

## 요구 사항

- Python >= 3.10
- Go >= 1.22 (3D 렌더러 사용 시)
- Rust (ONNX 추론 엔진 사용 시)
- nnUNet v2 모델 가중치 (세그멘테이션 실행 시)

## 설치

```bash
# 기본 설치
pip install -e .

# nnUNet 포함 설치
pip install -e ".[nnunet]"

# DICOM 지원 포함
pip install -e ".[dicom]"

# 개발 의존성 (pytest, ruff)
pip install -e ".[dev]"
```

## 실행

### Streamlit 앱

```bash
make run
# 또는
streamlit run src/medical_viewer/app.py --server.port=8501
```

브라우저에서 `http://localhost:8501`로 접속합니다.

### Go 3D 렌더러 (선택)

```bash
make run-renderer
# 또는
cd go_renderer && go run cmd/renderer/main.go
```

렌더러 없이도 Plotly 기반 3D 뷰어가 자동으로 사용됩니다.

### Docker

```bash
docker compose up
```

Streamlit(:8501)과 Go 렌더러(:8080)가 함께 실행됩니다.

## Rust 추론 엔진

PyTorch 대신 ONNX Runtime을 사용하여 빠른 추론을 수행합니다.

```bash
# 1. PyTorch 모델을 ONNX로 변환
python scripts/export_onnx.py \
    --model-dir /path/to/nnUNet_results/DatasetXXX/.../fold_all \
    --output-dir /path/to/output

# 2. Rust 추론 엔진 빌드
cd rust_inference
cargo build --release

# 3. 추론 실행
./target/release/nnunet-infer \
    --model model.onnx \
    --config preprocess_config.json \
    --input case.nii.gz \
    --output seg.nii.gz
```

## 설정

### `configs/app.yaml`

앱 포트, 렌더러 주소, 데이터 경로, nnUNet 가중치 디렉토리 등을 설정합니다.

### `configs/models.yaml`

세그멘테이션 모델 및 파이프라인을 정의합니다. 모델별 dataset ID, trainer, plans, 레이블 매핑 등을 지정합니다.

## 프로젝트 구조

```
medical-3d-viewer/
├── src/medical_viewer/
│   ├── app.py                 # Streamlit 메인 앱
│   ├── core/                  # 설정, 세션, DB, 내보내기
│   ├── inference/             # nnUNet 추론, 모델 레지스트리, 파이프라인
│   ├── mpr/                   # MPR 슬라이서, 윈도잉, 측정, Oblique
│   ├── renderer/              # Go 렌더러 HTTP 클라이언트
│   └── ui/                    # Streamlit UI 컴포넌트
├── go_renderer/               # Go 3D 렌더링 서버
│   ├── cmd/renderer/          # 서버 진입점
│   ├── internal/mesh/         # Marching Cubes, 메쉬 단순화
│   ├── internal/server/       # HTTP/WebSocket 서버
│   ├── internal/volume/       # 볼륨 로더
│   └── pkg/nifti/             # NIfTI 리더
├── rust_inference/            # Rust ONNX 추론 엔진
├── scripts/                   # ONNX 변환 스크립트
├── configs/                   # 앱 및 모델 설정
├── tests/                     # 테스트
├── docker-compose.yaml
├── Makefile
└── pyproject.toml
```

## 테스트

```bash
pip install -e ".[dev]"
pytest
```

## 라이선스

Private
