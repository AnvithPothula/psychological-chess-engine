# LatentBlunder, headless. Two stages: the build stage fetches Stockfish, the
# Python packages and the Maia-3 weights; the runtime stage carries only those
# and the code. The Lichess token is never in the image: pass it at run time
# (--env-file, see deploy/setup_systemd.sh).
#
#   docker build -t latentblunder .
#   docker run --rm --env-file bot.env -v "$PWD/build:/app/build" latentblunder

ARG PYTHON=3.13

FROM python:${PYTHON}-slim AS build
ARG TARGETARCH
ARG STOCKFISH_TAG=sf_19
ARG MAIA3_REF=main

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# Stockfish 19, the version the safety thresholds were tuned against, rather
# than Debian's older package. The universal build picks its CPU path at run
# time, so one image fits any VPS of the architecture.
RUN case "${TARGETARCH:-amd64}" in \
      amd64) arch=x86-64 ;; \
      arm64) arch=arm64 ;; \
      *) echo "no Stockfish build for ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
 && curl -fsSL "https://github.com/official-stockfish/Stockfish/releases/download/${STOCKFISH_TAG}/stockfish-linux-${arch}-universal.tar.gz" \
    | tar -xz -C /tmp \
 && binary="$(find /tmp -type f -name "stockfish-linux-${arch}*" ! -name '*.tar*' -perm -u+x | head -n 1)" \
 && install -m 0755 "${binary}" /usr/local/bin/stockfish \
 && echo uci | /usr/local/bin/stockfish | grep -q '^uciok$'

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH HF_HOME=/opt/hf
RUN python -m venv "${VIRTUAL_ENV}"

# Only what the bot imports. CPU torch first: Maia-3 asks for torch unpinned,
# so it is then satisfied instead of pulling the multi-gigabyte CUDA build.
COPY requirements.txt /tmp/requirements.txt
RUN torch_version="$(sed -n 's/^torch==\([^ ]*\).*/\1/p' /tmp/requirements.txt)" \
 && pip install --no-cache-dir "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cpu \
 && grep -E '^(berserk|chess|numpy|requests|certifi)==' /tmp/requirements.txt \
    | sed 's/[[:space:]]*#.*//' > /tmp/bot-requirements.txt \
 && pip install --no-cache-dir -r /tmp/bot-requirements.txt \
 && pip install --no-cache-dir "git+https://github.com/CSSLab/maia3@${MAIA3_REF}" \
 && maia3-cache --model maia3-5m


FROM python:${PYTHON}-slim

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH HF_HOME=/opt/hf PYTHONUNBUFFERED=1
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
COPY --from=build /usr/local/bin/stockfish /usr/local/bin/stockfish

RUN useradd --create-home --uid 10001 bot
WORKDIR /app
COPY src/ src/
COPY models/*.pth models/
RUN mkdir -p build && chown bot:bot build && chmod -R a+rX /opt/hf
USER bot

# Game log, bot log and caches. Mount it, or they go with the container.
VOLUME /app/build

# The bridge shuts down cleanly on Ctrl-C; SIGTERM would kill it mid-sentence.
STOPSIGNAL SIGINT
ENTRYPOINT ["python", "-m", "src.lichess.bot"]
CMD ["--allow-bots", "--challenge-bots", "--min-rating", "0", "--max-rating", "4000", \
     "--speeds", "blitz", "rapid", "classical"]
