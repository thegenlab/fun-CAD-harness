"""Explicit formal handoff limits; ordinary smoke defaults remain independent."""
PROMPT_BYTES = 1024 * 1024
WORK_BYTES = 512 * 1024 * 1024
SESSION_BYTES = 64 * 1024 * 1024
SESSION_METADATA_BYTES = 64 * 1024
METADATA_BYTES = 16 * 1024 * 1024
CONTENT_BYTES = WORK_BYTES + SESSION_BYTES + METADATA_BYTES
TAR_BYTES = CONTENT_BYTES + 16 * 1024 * 1024
GZIP_BYTES = TAR_BYTES + 1024 * 1024
CAPTURE_BYTES = 896 * 1024 * 1024
BASE64_CHUNK_BYTES = 49152
PUBLIC_INPUT_BYTES = 64 * 1024 * 1024
PUBLIC_REFERENCE_BYTES = WORK_BYTES + 64 * 1024 * 1024
PUBLIC_FILE_BYTES = 16 * 1024 * 1024
# The measured 4 GiB guest has a 2,057,990,144-byte tmpfs root. Leave room
# for its 341 MB client/bootstrap and about 106 MB of runtime files.
STAGED_BYTES = 1536 * 1024 * 1024
EVENT_BYTES = 8 * 1024 * 1024
FILE_COUNT = 4096
PATH_BYTES = 1024
# Allow the complete bounded base64 archive at 128 KiB/s plus 300 s to pack.
# This bounds post-model cleanup/export, not author work or Harness budget.
ENCODED_BYTES = ((GZIP_BYTES + 2) // 3) * 4
ENCODED_LINES = (GZIP_BYTES + BASE64_CHUNK_BYTES - 1) // BASE64_CHUNK_BYTES
POST_MODEL_CAPTURE_SECONDS = ((ENCODED_BYTES + ENCODED_LINES + 128 + 128 * 1024 - 1) // (128 * 1024)) + 300
# Each owner must outlive the inner save deadline: the controller decodes and
# validates the archive, then the workflow publishes work and session records.
FORMAL_CONTROLLER_STOP_SECONDS = POST_MODEL_CAPTURE_SECONDS + 300
FORMAL_WORKFLOW_STOP_SECONDS = FORMAL_CONTROLLER_STOP_SECONDS + 300


def validate_staged_bytes(*, work, session, reference, task, prompt, relocation=0):
    observed = 2 * work + 2 * session + reference + task + prompt + relocation
    if observed > STAGED_BYTES:
        raise ValueError(f'Formal guest staged data exceeds its bounded tmpfs allocation: {observed} > {STAGED_BYTES} bytes')
    return observed
