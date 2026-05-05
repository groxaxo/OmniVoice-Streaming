# DealerVoice and Open WebUI integration

This document records the local contract used to run OmniVoice as the TTS backend for:

- DealerVoice self-hosted: `/home/op/Videos/dealervoice-selfhosted-inworld-realtime-optional/dealervoice-selfhosted`
- Open WebUI: `/home/op/open-webui`

No secrets are required for this integration. Keep DealerVoice `infra/.env` and Open WebUI's persisted database private.

## Live ports

| Service | URL | Notes |
| --- | --- | --- |
| OmniVoice router | `http://127.0.0.1:6655` | GPU-aware router for the 3x RTX 3090 worker pool |
| OmniVoice app alias | `http://127.0.0.1:8081/v1` | `omnivoice-port8081.service`, safe for Open WebUI because Open WebUI owns `:8080` |
| OmniVoice Gradio | `http://127.0.0.1:7861` | Browser demo against the router |
| Open WebUI | `http://127.0.0.1:8080` | Configured to call OmniVoice on `:8081/v1` |
| DealerVoice API | `http://127.0.0.1:13000` | Docker API container; uses `host.docker.internal` for host services |

## DealerVoice TTS contract

DealerVoice's active local TTS path uses the Supertonic-compatible client:

```txt
LOCAL_PHONE_TTS_PROVIDER=supertonic
SUPERTONIC_TTS_BASE_URL=http://host.docker.internal:8081
SUPERTONIC_TTS_URL=http://host.docker.internal:8081
SUPERTONIC_TTS_MODEL=supertonic
SUPERTONIC_TTS_VOICE=M1
SUPERTONIC_TTS_TOTAL_STEPS=24
SUPERTONIC_TTS_SPEED=1.0
```

The client POSTs to `/v1/audio/speech` and sends:

```json
{
  "input": "Text to speak",
  "model": "supertonic",
  "voice": "M1",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false,
  "total_steps": 24
}
```

OmniVoice accepts that payload directly:

- `model=supertonic` is an alias for the OmniVoice backend model.
- `voice=M1` resolves through the default alias map to `monica`.
- `total_steps` is treated as `num_step` when `num_step` is absent.
- `lang_code` is treated as `language` when present.
- `stream=false`, `temperature`, `top_p`, `top_k`, and `repetition_penalty` are safely accepted/ignored so existing provider clients do not need changes.

## Open WebUI TTS contract

Open WebUI is configured as an OpenAI-compatible TTS client:

```txt
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://localhost:8081/v1
AUDIO_TTS_OPENAI_API_KEY=not-needed
AUDIO_TTS_MODEL=supertonic
AUDIO_TTS_VOICE=M1
```

Open WebUI uses `/audio/models`, `/audio/voices`, and `/audio/speech` below the configured base URL. OmniVoice exposes all of those under `/v1`.

## Smoke tests

Host-side DealerVoice/Open WebUI-compatible request:

```bash
curl -fsS -o /tmp/omnivoice-dealervoice.wav \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8081/v1/audio/speech \
  -d '{
    "model": "supertonic",
    "voice": "M1",
    "input": "Thanks for calling Valley Auto Group. This is Monica on OmniVoice.",
    "response_format": "wav",
    "speed": 1.0,
    "stream": false,
    "total_steps": 24
  }'
```

Open WebUI discovery:

```bash
curl http://127.0.0.1:8081/v1/audio/models
curl http://127.0.0.1:8081/v1/audio/voices
```

From the DealerVoice API container:

```bash
docker compose -f /home/op/Videos/dealervoice-selfhosted-inworld-realtime-optional/dealervoice-selfhosted/infra/docker-compose.yml \
  exec api node -e "fetch('http://host.docker.internal:8081/v1/audio/speech',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'supertonic',voice:'M1',input:'DealerVoice container smoke test.',response_format:'wav',total_steps:24,stream:false})}).then(async r=>{console.log(r.status,r.headers.get('content-type'),(await r.arrayBuffer()).byteLength)})"
```

Expected result: HTTP 200 and a WAV response.

## Operational notes

- Open WebUI must keep port `8080`; do not bind OmniVoice aliases there.
- `omnivoice-port8081.service` is a lightweight `socat` forwarder from `:8081` to the router on `:6655`.
- The main GPU pool config lives in `~/.config/omnivoice/omnivoice-pool.env`.
- The default DealerVoice voice alias map is:

```txt
m1=monica,default=monica,monicaoptimized=monica,monicaoptimized.wav=monica,monica-optimized=monica,monica_optimized=monica
```

Set `OMNIVOICE_VOICE_ALIASES` to override that map.
