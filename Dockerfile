FROM ghcr.io/taohunter251/reax-engine-v2@sha256:348257dd4e676dd92278304c0931fbfbf352b1150cbd3de3a8516ffbfce60181 AS v2
FROM ghcr.io/masterdev777/s1r@sha256:d57f7afa80e9e174132bdc5332e9064fd12de7bec6ba42f711a4752a3e36754a
ARG RUNTIME_SOURCE_REVISION=private-v3
ARG KERNEL_SOURCE_REVISION=private-v3
ARG ENGINE_HOTKEY=5EbyBZ4v86QH3JjGy7CxVwT1vEAezeqHaxJTP1Bn3Dfhqpjv
COPY --from=v2 --chown=s1m:s1m /opt/reax-v2/parameters /opt/reax-v3/parameters
COPY --chown=s1m:s1m vendor/s1eng /opt/s1eng
COPY --chown=s1m:s1m reaxv2 /opt/reax-v3/reaxv2
COPY --chown=s1m:s1m private /opt/reax-v3/private
COPY --chown=s1m:s1m runtime /opt/reax-v3/runtime
COPY --chown=s1m:s1m kernel-source /opt/reax-v3/kernel-source
COPY --chown=s1m:s1m NOTICE LICENSE parents.json construction-files.json /opt/reax-v3/
ENV PYTHONPATH=/opt/reax-v3/runtime:/opt/reax-v3:/opt/s1eng \
    PYTHONDONTWRITEBYTECODE=1
LABEL ai.reax.engine.hotkey="${ENGINE_HOTKEY}" \
      ai.reax.engine.ownership="[]" \
      ai.reax.engine.parents="[{\"uid\":56,\"digest\":\"23dcdecc1519edf813699ab48adadd1df07a269016da3816bee37cba84e8a89d\",\"files\":[\"/opt/s1eng/\",\"/opt/reax/\",\"/opt/reax-v3/\"]},{\"uid\":176,\"digest\":\"01e6e113da15bbc5f992779bb099a55d99b936ab088b64fb4967ef557f72db6d\",\"files\":[\"/opt/s1eng/\",\"/opt/reax/\",\"/opt/reax-v3/\"]},{\"uid\":164,\"digest\":\"944fa7aee414607c4b620de5600224fb7b844f0f9d844031b07f591b8b7455f7\",\"files\":[\"/opt/s1eng/\",\"/opt/reax/\",\"/opt/reax-v3/\"]},{\"uid\":240,\"digest\":\"d57f7afa80e9e174132bdc5332e9064fd12de7bec6ba42f711a4752a3e36754a\",\"files\":[\"/opt/s1eng/\",\"/opt/reax/\",\"/opt/reax-v3/\"]},{\"uid\":57,\"digest\":\"348257dd4e676dd92278304c0931fbfbf352b1150cbd3de3a8516ffbfce60181\",\"files\":[\"/opt/s1eng/\",\"/opt/reax/\",\"/opt/reax-v3/\"]}]" \
      ai.reax.engine.kernel-source="https://github.com/taohunter251/reax-engine-kernels-v3/tree/${KERNEL_SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/taohunter251/reax-engine-runtime-v3" \
      org.opencontainers.image.url="https://github.com/taohunter251/reax-engine-runtime-v3" \
      org.opencontainers.image.title="reax-engine" \
      org.opencontainers.image.version="v3" \
      org.opencontainers.image.revision="${RUNTIME_SOURCE_REVISION}" \
      org.opencontainers.image.description="Frozen private v3 with bucket-consistent projection policy and canonical suffix repairs"
ENTRYPOINT ["python", "/opt/reax-v3/runtime/bootstrap.py"]
