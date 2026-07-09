# VSS Search — WEKA App Store Blueprint

`vss-search.yaml` installs **VSS Search on WEKA** through the WEKA App
Store: an AI video search and alerting stack built on the NVIDIA Metropolis
VSS 3.1.0 search profile, adapted for multi-node Kubernetes with all state
on WEKA storage.

What you get after install:

- **Video ingest and recording** (VST) — register RTSP cameras, record to
  WEKA, replay and live-view from the browser (WebRTC via an in-cluster
  TURN relay; no host networking).
- **GPU perception** — DeepStream pipeline with GDINO open-vocabulary
  object detection; a stream distributor keeps every online camera assigned
  to a perception pod and self-heals after pod replacement.
- **Alert pipeline** — behavior analytics over Kafka, with each raw alert
  verified by the Cosmos Reason2 VLM before it appears as "Verified" in the
  UI.
- **AI agent + web UI** — chat over what the cameras have seen (backed by
  the Nemotron LLM and Cosmos VLM NIMs), alert review with thumbnails and
  clip playback, camera management.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| WEKA CSI driver + StorageClass | Install `csi-wekafsplugin` with its API secret first. The StorageClass must support `ReadWriteMany`; its name is the `storage_class` form field. |
| nginx ingress controller | The stack's Ingress uses `className: nginx`. If the controller is exposed on a NodePort, enter that port as `external_port` (e.g. `30080`). |
| GPU nodes | H100, L40S, or RTX PRO 6000 Blackwell. Minimum 5 GPUs: LLM NIM, VLM NIM, 2× perception, embedding service. NIMs require an exclusive GPU. |
| NGC credential | An NVIDIA NGC API key registered in the App Store (Settings → Credentials, type `nvidia-ngc`). Used to pull NIM/VSS images from nvcr.io and download model weights. |
| WEKA registry credentials | Your quay.io username and get.weka.io token, entered in the install form. Used to pull the WEKA-built images from `quay.io/weka.io/vss`. |

Registering the NGC credential from the CLI instead of the GUI:

```bash
kubectl create secret generic raw-ngc-key \
  --from-literal=NGC_API_KEY=<NGC_API_KEY> -n wekaappstore

kubectl apply -f - <<'EOF'
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: ngc-key
  namespace: wekaappstore
spec:
  type: nvidia-ngc
  displayName: "NVIDIA NGC API Key"
  secretRef:
    name: raw-ngc-key
    key: NGC_API_KEY
EOF
```

## Install form fields

| Field | Description |
|-------|-------------|
| `namespace` | Namespace the whole stack deploys into |
| `storage_class` | Pre-existing WEKA CSI StorageClass (RWX-capable) |
| `external_host` | Node IP or DNS name browsers use to reach the cluster |
| `external_port` | Ingress external port when not `:80` (e.g. `30080` for NodePort); empty otherwise |
| `gpu_type` | `H100`, `L40S`, or `RTXPRO6000BW` |
| `turn_password` | Shared secret for the WebRTC TURN relay (choose any strong value) |
| `weka_registry_user` | WEKA registry username for quay.io |
| `weka_registry_token` | WEKA registry token (get.weka.io) for quay.io |
| `ngc_credential` | The registered nvidia-ngc credential |

## What the blueprint deploys

Three components, in dependency order:

1. **vss-values** — a ConfigMap holding the stack's Helm values, rendered
   from the form fields.
2. **nim-operator** — the NVIDIA NIM Operator (from the NGC Helm
   repository), which manages the LLM/VLM model caches and services.
3. **vss-search** — the VSS Search umbrella chart
   (source: <https://github.com/weka/neuralmesh-vss3-k8s>), with values
   loaded from the ConfigMap.

Container images come from `nvcr.io` (authenticated via the NGC
credential) and `quay.io/weka.io/vss` (authenticated via your WEKA
registry credentials, stored as the `weka-quay-secret` pull secret).

## Monitoring the install

```bash
kubectl get wekaappstore vss-search -n <namespace> -w

kubectl get wekaappstore vss-search -n <namespace> \
  -o jsonpath='{range .status.componentStatus[*]}{.name}{"\t"}{.phase}{"\t"}{.message}{"\n"}{end}'

helm list -n <namespace>
```

Readiness is reported when the web UI is serving. Model caching continues
in the background — the first chat and alert-verification responses may lag
until the NIM caches finish downloading (20+ minutes for H100-class
weights):

```bash
kubectl get nimcaches -n <namespace>
```

Once ready, open `http://<external_host>:<external_port>/` and add cameras
under **Video Management** (RTSP URLs), or register them via the VST API.

## Post-install tuning

Two knobs are commonly adjusted per site after install (via the release
values):

- `perception.gdino.typeName` — the open-vocabulary detection prompt, e.g.
  `"person . vehicle . ;0.4"`. Set it to the object classes that matter at
  your site.
- `behavior-analytics-alerts.config.fovObjectType` — the object class that
  field-of-view alerts fire on.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Stack phase `Failed` immediately | `kubectl describe wekaappstore vss-search -n <ns>` — conditions list the failing component and reason |
| Pods `ErrImagePull` on quay.io images | The `weka-quay-secret` in the app namespace holds the registry credentials from the form; verify the username/token are valid for quay.io (get.weka.io) |
| Pods `ErrImagePull` on nvcr.io images | The credential's derived pull secret must exist in the app namespace: `kubectl get secret <warp-...-docker> -n <ns>`. Verify the credential shows `KeyReady: True` in Settings → Credentials. |
| NIMs stuck downloading | `kubectl get nimcaches -n <ns>`; the NGC key must be valid for the NIM models used |
| Live View shows no video | The TURN relay runs at `<external_host>:3478`; verify the port is reachable from the browser |
