# SiliconFlow Official VLM Defaults Design

**Goal:** Align the OCR implementation with SiliconFlow's current official multimodal chat completions guidance by updating default endpoint and model settings, and by exposing image detail control for OCR quality tuning.

## Scope

- Update the default SiliconFlow base URL to `https://api.siliconflow.cn/v1`.
- Update the default OCR model to `Qwen/Qwen3-VL-8B-Instruct`.
- Add a configurable image detail setting passed through the existing `image_url` payload.
- Keep the public OCR processor API and pipeline wiring unchanged.

## Request Shape

The OCR request continues to use SiliconFlow's OpenAI-compatible `chat.completions.create(...)`
entry point with a single user message containing:

- a text prompt
- an `image_url` content item
- `image_url.detail` set from configuration

This preserves the current integration pattern while matching the official multimodal request shape
more closely for OCR workloads.

## Configuration

`VLMSettings` should gain an `image_detail` field sourced from `SILICONFLOW_IMAGE_DETAIL`.

- Allowed values: `auto`, `low`, `high`
- Default: `high`

Invalid values should raise a `ValueError` during settings loading so configuration issues fail
early and clearly.

## Compatibility

- `OCRProcessor` constructor stays unchanged.
- Existing prompt, retry, timeout, and rate-limiting behavior stays unchanged.
- Existing callers that rely on environment overrides remain compatible.

## Testing

Tests should cover:

- new default base URL and model values
- default image detail value
- invalid `SILICONFLOW_IMAGE_DETAIL` rejection
- request payload including `image_url.detail`

## Documentation

Update `.env.example` and `README.md` to reflect:

- the new default base URL
- the new default OCR model
- the new `SILICONFLOW_IMAGE_DETAIL` setting
