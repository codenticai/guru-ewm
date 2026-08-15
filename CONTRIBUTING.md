# Contributing

Thanks for your interest in Guru-EWM.

## Getting started

1. Fork and clone the repository.
2. Copy `.env.example` to `.env` and adjust the build contexts:
   ```bash
   cp .env.example .env
   ```
3. Build and run:
   ```bash
   docker compose build
   docker compose up -d
   ```
4. Run the tests (requires the stack to be up for the API/GUI suites):
   ```bash
   pip install -r tests/requirements-test.txt
   pytest tests/ -v
   ```

## Making changes

- Keep changes small and focused.
- Follow the existing code style in each service.
- Update the relevant docs under `docs/` when behavior changes.
- Add or update tests for bug fixes and new features.

## Areas

| Area | Location |
|---|---|
| NLP model | `nlp-model/` |
| Medical diagnostic | `medical-diagnostic/` |
| OCR | `deepseek-ocr/` |
| Gateway | `ewm-gateway/` |
| UI | `ewm-ui/` |
| Tooling/scripts | `scripts/` |
| Tests | `tests/` |

## Submitting

- Open a pull request with a clear description.
- CI builds the repo-local services and validates the compose file.
- Maintainers will review; please be patient.

Thank you!
