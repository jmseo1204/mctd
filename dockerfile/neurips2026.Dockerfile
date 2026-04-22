FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HOME=/tmp
ENV TEXMFVAR=/tmp/texmf-var
ENV TEXMFCONFIG=/tmp/texmf-config
ENV TEXMFHOME=/tmp/texmf-home

RUN apt-get update && apt-get install -y --no-install-recommends \
  latexmk \
  make \
  perl \
  ghostscript \
  texlive-latex-base \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-pictures \
  texlive-plain-generic \
  texlive-science \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/texmf-var /tmp/texmf-config /tmp/texmf-home

WORKDIR /workspace/paper/neurips2026

CMD ["bash"]
