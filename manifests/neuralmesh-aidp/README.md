# WEKA AIDP Blueprint

Deploys the full AIDP platform (Keycloak, Milvus, the NVIDIA RAG Blueprint +
NIMs, the LGTM+Phoenix observability stack, and the AIDP services) into the
`rag` namespace via the WEKA App Store operator.

Generated from `appstack/weka-aidp-appstack.yaml` in the `aidp` repo, converted
to a real App Store blueprint: `x-variables` install form, Jinja2 `[[ ]]`
rendering, and the credential feature.

## Before you install — register credentials (Settings page)

The install form will not enable until these exist:

1. **NVIDIA NGC** (`type: nvidia-ngc`) — the NGC API key. The blueprint copies
   the derived `warp-<name>-docker` / `warp-<name>-apikey` secrets into `rag` as
   `ngc-secret` and `ngc-api` (the names nv-ingest hardcodes).
2. **WEKA storage** (`type: weka-storage`) — the WEKA REST API token. Used to
   set `aidp-gui-operator-secrets.WATCHER_WEKA_TOKEN` for the AIDP operator.

## Install-form fields (`x-variables`)

| Field | Purpose |
|---|---|
| `ngc_credential` | NVIDIA NGC credential (dropdown) |
| `weka_credential` | WEKA storage credential (dropdown) |
| `weka_s3_access_key` / `weka_s3_secret_key` | WEKA S3 keys for Milvus + nv-ingest (not covered by the weka-storage credential type) |
| `weka_s3_node_ips` | Comma/space-separated WEKA S3 node IPs for the nginx S3 LB |
| `weka_api_host` | WEKA cluster management IP (REST API, port 14000) |
| `keycloak_fqdn` | External Keycloak FQDN |
| `gui_base_url` | External AIDP GUI URL (OIDC redirect derived from it) |
| `attu_hostname` | Milvus Attu admin UI FQDN |
| `cluster_fqdn` | Cluster domain (Phoenix host = `phoenix.<domain>`) |
| `vectordb_filesystem_name` | WEKA filesystem name backing the vector DB |

## Notes

- The two substitution layers: the GUI resolves `[[ ]]` at install time; the
  operator resolves only `${wekaS3AccessKey}` / `${wekaS3SecretKey}` at deploy
  time (allowlist substitution) and leaves all bash `${...}` untouched. Bash
  scripts that use `[[ ... ]]` are wrapped in `{% raw %}` so the GUI's Jinja2
  does not treat them as variables.
- Internal app secrets (Keycloak admin / Postgres passwords, `SESSION_SECRET`,
  `TOOL_API_KEY`) retain the source defaults and are not exposed on the form.
- The GPU profile baked into `aidp-site-config` is **4x RTX PRO 6000 SE**. Edit
  the `nvidia-gpu-profile` key for other hardware (profiles list in the AIDP
  repo `helm/aidp/profiles/`).
