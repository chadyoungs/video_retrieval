# Video Retrieval with ActivityNet / nuScenes

Hybrid (dense + structured) retrieval over video segments: CLIP/SigLIP embeddings in
Milvus, annotations and metadata in PostgreSQL, and a rule + cross-encoder + VLM
reranker on top.

## Layout

| Path | Role |
| --- | --- |
| `config/settings.py` | single source of runtime config, env / `.env` overridable |
| `datasets/` + `datasets/builders/` | per-dataset discovery (`VideoEntry`) and the nuScenes clip builder |
| `annotators/` | Ollama VLM annotation per dataset, prompt + JSON schema + validation |
| `encoders/` | keyframe encoder, text encoder, video decoding, keyframe selection |
| `ingestion/` | segment -> annotate -> embed -> persist pipeline and its CLI |
| `db/` | Milvus + PostgreSQL stores behind one interface, one table/collection per dataset |
| `rerankers/` | integrated reranker (rule gate, cross-encoder, VLM) |
| `retrieval/` | hybrid recall + fusion + rerank service and its CLI |
| `utils/` | device resolution, HF model cache helpers, legacy adapter |
| `alembic/` | schema migrations |
| `models_download/` | weight downloader that matches the cache layout |
| `tests/` | offline checks (no Milvus / PG / model weights required) |

## Requirements

- Python **3.10** (tested with 3.10; 3.12 also works for the non-torch parts)
- PostgreSQL with the `pg_dsn` database reachable
- Milvus (standalone or cluster), default `localhost:19530`; [Attu](https://github.com/zilliztech/attu) is handy for inspection
- Ollama serving a vision model for annotation / VLM rerank (`ollama pull qwen3-vl:8b`)
- CUDA GPU recommended (RTX 3060 Ti 8G is enough for CLIP base + keyframes)

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit paths / DSN / endpoints
```

Run every command **from the repository root** (`python -m ...` resolves imports from
the current directory).

## Model weights

```bash
python -m models_download.download --preset clip           # openai/clip-vit-base-patch32

python -m models_download.download --repo-id <hf-repo-id>  # anything else
```

Weights land in `MODEL_CACHE_DIR/<repo basename>` and are reused by the encoders and
the cross-encoder. Set `HF_LOCAL_ONLY=true` on an offline box to forbid downloads.

## Storage setup

```bash
alembic upgrade head          # canonical schema: activitynet_video_segments, nuscenes_video_segments
```

`IngestionPipeline(...).init()` also creates the table / collection on a fresh dev
database, and `--drop-sql --drop-milvus` recreates them.

## Ingestion

```bash
# ActivityNet: all videos under ACTIVITYNET_DATA_ROOT, sliced by CLIP_DURATION_SEC
python -m ingestion.main --dataset activitynet --limit 20

# rebuild indexes from scratch
python -m ingestion.main --dataset activitynet --drop-sql --drop-milvus

# nuScenes (requires the clip mapping from the builder, see below)
python -m ingestion.main --dataset nuscenes
```

`--dataset` selects the dataset adapter, the PostgreSQL table **and** the Milvus
collection, so a run can never mix the two datasets.

Per video the pipeline slices overlapping windows (`CLIP_DURATION_SEC`,
`CLIP_OVERLAP_SEC`), samples `FRAME_SAMPLING_RATE` fps, annotates the keyframes with
the VLM, embeds them with the keyframe encoder, and writes one row per segment:

- **PostgreSQL**: `segment_id`, clip/timing, `dataset_version/split`, annotation
  fields, `annotation_model`, `processing_ts`.
- **Milvus**: `segment_id` (primary key), display fields, `dataset_name`, `clip_id`,
  `camera_channel` (nuScenes), `embedding_model`, `clip_vector`.

`segment_id = <dataset>_<clip>_<start_ms>_<end_ms>` is the join key between both
stores and the identity used during retrieval. If the VLM call fails or its output
fails validation, annotation falls back to `unknown` values. A segment with no
readable frames is skipped and counted in `skipped_segments` instead of aborting the
run.

### nuScenes clip builder

```bash
python -m datasets.builders.nuscenes_video_builder \
    --dataroot <NUSCENES_DATA_ROOT> \
    --meta-root <NUSCENES_META_JSON_ROOT> \
    --output-dir <OUTPUT_VIDEO_DIR> \
    --dataset-version v1.0-mini --dataset-split train --fps 12
```

Writes CAM_FRONT MP4 clips plus `NUSCENES_CLIP_MAPPING_FILE` (clip-to-source traceability:
scene/log/location/map tokens, sample range, ego speed, frame count). Note the clips are
already ~`CLIP_DURATION_SEC` long and the pipeline slices them again - keep the builder
duration >= the ingestion window, or set `CLIP_OVERLAP_SEC=0` for pre-cut clips.

## Retrieval

```bash
# text
python -m retrieval.main --type text --query "a man playing guitar" --limit 5

# annotation conditions (OR) + metadata filters (AND)
python -m retrieval.main --type text --query "busy street with pedestrians" \
    --annotation-conditions '{"scene_env": "outdoor", "people_count": "crowd"}' \
    --metadata-filters '{"dataset_name": "activitynet"}'

# image / video query
python -m retrieval.main --type image --query ./samples/frame.png --limit 5
python -m retrieval.main --type video --query ./query_clip.mp4 --limit 5

# inspect the coarse order, or emit machine-readable output
python -m retrieval.main --type text --query "office" --no-rerank --json

# only segments whose annotation passed the quality policy
python -m retrieval.main --type text --query "busy street" --only-good
```

### How a query is served

1. **Recall.** Milvus ANN (`pool = max(limit, RERANKER_TOP_K) * RECALL_POOL_MULTIPLIER`)
   plus a PostgreSQL query that returns segments satisfying the annotation / metadata
   conditions, scored by `matched_conditions / condition_count` (1.0 when only filters
   are given). With no annotation conditions and no filters the relational side
   contributes nothing - an unscored full-table scan would inject arbitrary rows.
2. **Fusion** per `segment_id`: `final = 0.7 * vec + 0.3 * sql`. A candidate present on
   only one side keeps `0` for the other, never a perfect match. Vector scores are
   cosine similarity mapped from `[-1, 1]` to `[0, 1]`.
3. **Annotation back-fill.** Vector-only candidates are looked up by `segment_id` so the
   reranker always has annotation text.
4. **Hard metadata filter.** Applied after fusion, which also covers fields Milvus does
   not store (annotation columns, `dataset_version`, ...). Unknown fields are skipped
   with a log line instead of silently dropping everything.
5. **Rerank** (skip with `--no-rerank`):
   - *rule gate*: `metadata_filters` AND, hard drop;
   - *rule score*: `annotation_conditions` OR-match ratio (only when conditions exist);
   - *cross-encoder*: query vs annotation text, sigmoid score;
   - *VLM*: query (text or image) plus one frame extracted from the candidate segment,
     asked for a 0-10 score.
   Missing components (weights absent, Ollama down, no query text) are **skipped and
   the remaining weights renormalized**; with no signal at all the coarse order is
   kept. Each result carries `segment_id`, `clip_id`, `quality_flag`, `score_rule`,
   `score_cross_encoder`, `score_vlm` and a `rerank_reason` string.

### Annotation quality flag

Every ingested segment carries the existing `quality_flag`, computed right after
annotation and before the row is written (`annotators/quality.py`):

- **Enum fields** (`scene_env`, `scene_type`, `people_count`, `lighting`, `time_of_day`):
  a value that resolves to `unknown` (`unknown`/`unk`/`none`/`n/a`/`""`/`unclear` ...)
  or falls outside the allowed vocabulary makes the segment `bad`.
- **`semantic_summary`**: missing, empty, `unknown`, mojibake (`\ufffd` or control
  characters), a bare placeholder (`[...]`), longer than 512 chars, no letters/digits,
  more than 30% noise symbols, a single repeated letter, or one token repeated three
  times all make the segment `bad`.
- **Meaningless short phrases**: a summary built only from media/filler vocabulary
  (`unknown scene`, `a video clip`, `no information available`, `some objects`,
  `nothing is visible`) or one that still hedges (`unknown man walking`,
  `the footage is unclear`) is `bad` too; so is anything with fewer than two content
  words (`car`, `street`). Chinese one-liners are compared against the same filler
  list, so `未知` / `没有内容` / `街道` are rejected while real sentences pass.
  The hedging test is part of `QUALITY_STRICT_UNKNOWN`.

The rows are still written; bad segments are counted in the ingestion summary
(`bad_quality_segments`, `quality_reasons`, one log line per reason up to a budget,
then an aggregate), so a suspicious run is visible immediately.

Four knobs in `.env`:

```bash
QUALITY_FLAG_OK=ok
QUALITY_FLAG_BAD=bad                  # or not_ok - anything that fits VARCHAR(16)
QUALITY_STRICT_UNKNOWN=true           # false = tolerate every unknown enum + hedging text
QUALITY_UNKNOWN_IGNORE_FIELDS=""      # csv, e.g. time_of_day,lighting
```

`quality_flag` is not in the Milvus schema, so filtering happens on the PostgreSQL
side and in the post-fusion hard filter; the value is echoed in every result.
`--only-good` is shorthand for
`--metadata-filters '{"quality_flag": "<QUALITY_FLAG_OK>"}'`.

## Offline checks

These need neither Milvus, PostgreSQL, CLIP weights nor Ollama:

```bash
python -m tests.test_retrieval_offline   # fusion / filters / rerank / loader / SQL
python -m tests.test_ingestion_offline   # pipeline rows, keys, buffering, --limit
python -m tests.test_quality_flag        # quality policy only, pure functions
```

They cover: segment identity (two segments of one clip must not share scores),
annotation back-fill, rule scoring and gating, hard metadata filters on fields Milvus
cannot filter, SQL candidate recall, deterministic relational ranking
(`search_metadata` returning nothing when unscored), `get_segments_by_ids`, real
SQLAlchemy query generation (against a temporary SQLite file), frame sampling by
timestamp, the ingestion row contract, the quality policy (unknown enums, garbage
summaries, persisted `quality_flag`, adapter metadata not being able to fake `ok`),
and feature-output normalisation (tensor / `text_embeds` / `pooler_output`).

They do **not** cover live ANN behaviour, PG-specific DDL, or real model output - run
one small ingestion + retrieval pass after pulling, as in the checklist below.

## Appendix

### Text encoder feature shapes

`transformers` does not always return a tensor from `get_text_features` /
`get_image_features`; some versions and checkpoint families return a model output
object (`BaseModelOutputWithPooling`, `CLIPTextModelOutput`). `utils/features.py`
normalises both (`text_embeds` -> `image_embeds` -> `pooler_output` -> mean-pooled
`last_hidden_state`), so encoders never call `.detach()` on the wrong object.

If the text encoder then reports `produced N-d features but the index expects M-d`,
the checkpoint is a text-only encoder (pooler output is the transformer hidden size,
e.g. 768 instead of 512) - load the multimodal checkpoint
(`openai/clip-vit-base-patch32` or `google/siglip-so400m-patch14-384`) and keep
`SEMANTIC_MODEL_TIER` aligned with the dimension of the existing index.

### Hugging Face download behind the GFW

```bash
pip install -U huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT="https://hf-mirror.com"
hf download google/siglip-so400m-patch14-384
```

`config/settings.py` also sets `HF_ENDPOINT` from `HF_ENDPOINT` for in-process loads.

### CUDA

Encoders and the cross-encoder move to the resolved device (`DEVICE=auto` picks CUDA,
then MPS, else CPU). `DEVICE=cuda` fails loudly when CUDA is unavailable instead of
silently running on CPU.
