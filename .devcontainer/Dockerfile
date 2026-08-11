FROM mcr.microsoft.com/devcontainers/base:trixie
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
