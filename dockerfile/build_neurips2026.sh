#!/usr/bin/env bash
set -euo pipefail

docker build -f dockerfile/neurips2026.Dockerfile -t neurips2026 .
