# Open WebUI Add-on Blueprint

Deploys Open WebUI as an add-on to an already-running AIDP cluster — an
OIDC-authenticated chat UI backed by the AIDP RAG pipeline via `rag-bridge`.

Generated from `appstack/weka-openwebui-appstack.yaml` in the `aidp` repo.

## Prerequisite

The **WEKA AIDP** blueprint must already be installed and Ready. This add-on
reuses AIDP's Keycloak, `tool-api-key` Secret, and `nim-llm`. It registers its
own `open-webui` OIDC client in the `rag-gateway` realm and derives the OIDC
provider URL, secrets, and Envoy hostAlias from existing AIDP ConfigMaps at
deploy time.

No credentials are required for this add-on (AIDP already materialized them).

## Install-form fields (`x-variables`)

| Field | Purpose |
|---|---|
| `openwebui_base_url` | External Open WebUI URL. The OIDC redirect URI and the Envoy HTTPRoute hostname are derived from it. |

Bash Job scripts that use `[[ ... ]]` are wrapped in `{% raw %}` so the GUI's
Jinja2 (`[[` = variable start) renders them verbatim.
