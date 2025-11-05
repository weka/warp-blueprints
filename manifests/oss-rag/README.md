# OSS RAG Stack (Kubernetes + Helm Operator)

This blueprint defines an end‑to‑end Retrieval‑Augmented Generation (RAG) stack using open‑source components. It deploys a vector database, model serving for both chat and embeddings (via vLLM), and a user‑friendly web UI. The stack is expressed as a single Kubernetes custom resource so a Helm operator can orchestrate the underlying Helm charts and manifests in the correct order.

If you’re new to these concepts:
- RAG (Retrieval‑Augmented Generation) improves AI responses by first retrieving relevant documents from a knowledge base and then feeding them to a language model to ground its answers in your data.
- vLLM is a high‑performance server for running open models behind an OpenAI‑compatible API.
- Milvus is a vector database optimized for storing and searching embeddings (numerical representations of text/documents).
- Open WebUI is a simple web app that lets you chat with the model and upload documents.

The blueprint file is `oss-rag-stack.yaml` and the Custom Resource kind is `WekaAppStore`. An operator reconciles this resource and installs each component in dependency order.

---

## What this stack deploys

1) cert-manager
- Purpose: Manages TLS certificates inside the cluster.
- Why it’s here: Several charts benefit from cert-manager’s CRDs (and it’s common in production clusters).

2) Milvus (Vector Database)
- Purpose: Stores embeddings and supports fast similarity search.
- Notes:
  - Cluster mode is enabled.
  - Includes Attu (Milvus UI) optionally for exploration.
  - Service is exposed as a ClusterIP within the `default` namespace.

3) Milvus Init Job
- Purpose: Creates a logical Milvus database named `openwebui` used by the UI.
- Notes: Implemented as a Kubernetes Job using `curl` against Milvus’ HTTP API. It’s idempotent (safe to run multiple times).

4) vLLM Embedding Server
- Purpose: Serves an embedding model (e.g., `intfloat/e5-mistral-7b-instruct`) over an OpenAI‑compatible API.
- Notes:
  - Requires GPU resources.
  - Exposes a Kubernetes Service used by Open WebUI for document embedding.

5) vLLM Chat Server
- Purpose: Serves a chat/instruction‑tuned model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) over an OpenAI‑compatible API.
- Notes:
  - Requires GPU resources.
  - Exposes a Kubernetes Service consumed by Open WebUI for chat.

6) Hayhooks (placeholders)
- Purpose: Example slots for a Haystack/Hayhooks indexing pipeline and RAG search service.
- Notes:
  - Disabled by default and represented as placeholders. Replace with your manifests or charts if you use Hayhooks.

7) Open WebUI
- Purpose: Frontend for chatting with the model and uploading documents.
- Notes:
  - Configured to use Milvus as the vector store.
  - Points to the vLLM Chat and Embedding services via OpenAI‑compatible endpoints.
  - Persistence enabled by default (PVC) for uploaded files/config.

---

## How the pieces work together (RAG flow)

1. You upload documents in Open WebUI.
2. Open WebUI calls the Embedding Server (vLLM embedding) to convert text into embeddings.
3. The embeddings are stored in Milvus under the `openwebui` database.
4. When you ask a question, Open WebUI retrieves the most relevant chunks from Milvus.
5. Those chunks are sent along with your question to the Chat Server (vLLM chat) to produce a grounded answer.

This “retrieve → augment → generate” pattern helps the LLM provide accurate, up‑to‑date responses based on your data.

---

## File of record

- `oss-rag-stack.yaml`: Single source of truth for the whole stack. It includes:
  - A `WekaAppStore` spec listing each component, its Helm chart or manifest, dependency order (`dependsOn`), readiness checks, and target namespace.
  - Embedded ConfigMaps with Helm values for Milvus, vLLM (chat + embed), and Open WebUI.

The operator reads this resource and ensures components are installed in order:
- cert-manager → Milvus → Milvus Init Job → vLLM (embed + chat) → Open WebUI

---

## Prerequisites

- A Kubernetes cluster with:
  - NVIDIA GPUs available for the vLLM components (1 GPU per vLLM server as configured). Adjust resources if you have different hardware.
  - A default StorageClass for PVCs.
- A Helm operator/controller that understands the `WekaAppStore` CRD and can install the referenced Helm charts. (This blueprint assumes such an operator is present in the cluster.)
- Optional: A valid `HUGGING_FACE_HUB_TOKEN` stored in a Secret named `hf-token-secret` (namespace `default`). The vLLM pods reference this secret to pull models.

---

## Important configuration highlights

- Milvus
  - Cluster mode enabled, service at `http://my-release-milvus.default.svc.cluster.local:19530`
  - Database created via the init Job: `openwebui`
  - Prometheus ServiceMonitor support enabled (namespace `monitoring`, label `release=kube-prom-stack`)

- vLLM Embedding
  - Model: `intfloat/e5-mistral-7b-instruct`
  - Requests: 1 GPU, ~10Gi memory, 1 CPU (adjust per your nodes)
  - Exposes an OpenAI‑compatible `/v1` endpoint via a Service named like `vllm-embed-mistral-7b-engine-service` in `default`

- vLLM Chat
  - Model: `mistralai/Mistral-7B-Instruct-v0.3`
  - Requests: 1 GPU, ~5Gi memory, 1 CPU
  - Exposes an OpenAI‑compatible `/v1` endpoint via a Service named like `vllm-chat-mistral-7b-chat-engine-service` in `default`

- Open WebUI
  - Uses Milvus (`VECTOR_DB=milvus`, `MILVUS_URI`, `MILVUS_DB=openwebui`)
  - Points to vLLM chat and embedding endpoints via `OPENAI_API_BASE_URL` and `RAG_OPENAI_API_BASE_URL`
  - Signup and login enabled by default; name set to "WEKA Enterprise Search (RAG)"

- Placeholders
  - `hayhooks-indexing` and `hayhooks-rag-search` are disabled placeholders—replace their manifests/values and set `enabled: true` if you use them.

---

## Deploying the stack

Because this is a `WekaAppStore` resource, deployment is typically:

1. Ensure the Helm operator and CRDs are installed and running in your cluster.
2. Create required secrets (optional but recommended):
   - `hf-token-secret` in `default` with key `token` containing your Hugging Face token.
3. Apply the blueprint:
   ```bash
   kubectl apply -f oss-rag-stack.yaml
   ```
4. Watch components come up in order (the operator respects `dependsOn` and readiness checks):
   ```bash
   kubectl get pods -n default -w
   ```
5. Access Open WebUI (ClusterIP by default):
   - Port-forward:
     ```bash
     kubectl -n default port-forward svc/openwebui 3000:3000
     ```
     Then open http://localhost:3000
   - Or enable an Ingress in the Open WebUI values if you need external access.

---

## Customizing

- Models and resources
  - Edit the `vllm-config` ConfigMap (both `vllm-stack-embed.yaml` and `vllm-stack-chat.yaml`) to change the model, GPU/CPU/memory, storage, or node selectors.
- Vector database settings
  - Modify the `milvus-config` ConfigMap values (e.g., persistence, metrics, Attu, service type).
- Open WebUI behavior
  - Adjust environment variables in the `openwebui-config` ConfigMap to switch vector DBs, change endpoints, or branding.
- Ingress & exposure
  - Enable and configure Ingress for Open WebUI and any other services (Milvus Attu, etc.) per your cluster’s ingress controller.

After editing the ConfigMaps within `oss-rag-stack.yaml`, re-apply the file. The operator should roll out the updated configuration.

---

## Troubleshooting

- Pods Pending (GPU)
  - Ensure nodes have GPUs and the scheduler can place the vLLM pods. Check node selectors and runtime class.
- Model download issues
  - Confirm `hf-token-secret` is present and accessible in `default`.
- Open WebUI cannot connect to Milvus or vLLM
  - Verify the in‑cluster service DNS names from this README match the Services created by the operator.
  - Use `kubectl port-forward` and curl the `/v1/models` endpoint on the vLLM services to test reachability.
- Cert-manager readiness
  - This component must be ready before other chart installs that rely on its CRDs.

---

## Security notes

- The example uses a dummy `OPENAI_API_KEY` because vLLM does not require one by default; it simply expects the header for compatibility. Replace with a proper secret management approach in production.
- Consider enabling persistence for Milvus and hardening network policies, RBAC, and Ingress TLS for production use.

---

## Learn more

- Milvus Helm chart: https://zilliztech.github.io/milvus-helm/
- vLLM Production Stack: https://vllm-project.github.io/production-stack/
- Open WebUI Helm chart: https://open-webui.github.io/helm-charts
- Milvus: https://milvus.io/
- vLLM: https://vllm.ai/
- Open WebUI: https://github.com/open-webui/open-webui
