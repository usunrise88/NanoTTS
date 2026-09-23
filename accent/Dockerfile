FROM python:3.12-slim

# transformers is pinned because RUAccent's ONNX graphs expect token_type_ids,
# which newer tokenizers stop emitting; the model then fails at runtime.
RUN pip install --no-cache-dir \
        ruaccent==1.5.8.3 \
        transformers==4.57.6 \
        onnxruntime==1.23.2 \
        huggingface-hub

COPY server.py /app/server.py

# Models are baked in at build time rather than fetched on boot. RUAccent also
# writes Koziev's rule engine into its own package directory, which a non-root
# runtime user cannot do — and a service that needs the network to start is a
# service that fails when the network does.
ENV RUACCENT_WORKDIR=/opt/ruaccent
RUN python -c "import sys; sys.path.insert(0,'/app'); import server; server.prefetch()" \
    && chmod -R a+rX /opt/ruaccent /usr/local/lib/python3.12/site-packages/ruaccent

RUN useradd -r -u 10002 -m -d /home/accent accent
USER accent

ENV PORT=8100 RUACCENT_POOL=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=3s --start-period=90s \
  CMD ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8100/healthz').read()"]
ENTRYPOINT ["python", "/app/server.py"]
