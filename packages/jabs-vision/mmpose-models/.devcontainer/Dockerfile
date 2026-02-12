FROM ghcr.io/kumarlabjax/mmpose-base:latest

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# Create a non-root user and fail loudly if it already exists in base image
RUN set -eux; \
    if getent group "${USERNAME}" >/dev/null; then \
        echo "Group ${USERNAME} already exists in base image; aborting" >&2; \
        exit 1; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        echo "User ${USERNAME} already exists in base image; aborting" >&2; \
        exit 1; \
    fi; \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    useradd -m -s /bin/bash --uid "${USER_UID}" --gid "${USER_GID}" "${USERNAME}"; \
    mkdir -p /workspace; \
    chown -R "${USERNAME}:${USERNAME}" /workspace

# let the user run apt if you ever need it interactively
RUN apt-get update \
 && apt-get install -y --no-install-recommends sudo \
 && rm -rf /var/lib/apt/lists/* \
 && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
 && chmod 0440 /etc/sudoers.d/vscode

USER ${USERNAME}
