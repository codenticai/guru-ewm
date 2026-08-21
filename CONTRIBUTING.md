# Contributing

Thanks for your interest in Guru-EWM.

## Getting started

1. Fork and clone the repository.
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Build and run:
   ```bash
   docker compose build
   docker compose up -d
   ```

## Making changes

- Keep changes small and focused.
- Follow the existing code style in each service.

## Areas

| Area | Location |
|---|---|
| NLP model | `nlp-model/` |
| Medical diagnostic | `medical-diagnostic/` |
| OCR | `deepseek-ocr/` |
| Gateway | `ewm-gateway/` |
| UI | `ewm-ui/` |

## Submitting

- Open a pull request with a clear description.
- CI builds the repo-local services and validates the compose file.
- Maintainers will review; please be patient.

Thank you!
