# OCEANMES connection framework

## Scope of this slice

The Jetson is the HTTP client. It retrieves authoritative device configuration
and submits one labeled JPEG plus a versioned inspection manifest:

~~~text
GET  /api/edge/v1/config
POST /api/edge/v1/inspections
Authorization: Bearer oce_edge_<secret>
~~~

The first implementation provides:

- strict environment-based connection settings;
- HTTPS-by-default transport with bounded connect/read timeouts;
- bearer-key authentication without logging or displaying the key;
- parsing of the server's authoritative room, line, and configuration version;
- construction of payload-version-1 manifests;
- streaming SHA-256 calculation for the model and evidence;
- multipart upload with exactly the manifest and evidence parts; and
- structured errors that distinguish retryable transport/server failures from
  permanent contract or authentication failures.

It does not yet call the client from the inference thread. The scheduler,
representative-frame selection, dual-camera composite, and durable retry outbox
belong to the next integration slice.

## Runtime boundary

~~~text
camera capture -> inference -> inspection result + labeled JPEG
                                      |
                                      v
                         future bounded handoff/outbox
                                      |
                                      v
                     OCEANMES sync worker -> OceanMesClient
~~~

Model hashing must happen once at process startup. Evidence encoding, hashing,
disk persistence, and network upload must occur outside the inference hot path.
No network failure may pause camera capture, inference, or solenoid safety.

## Device configuration

Provision the device from OCEANMES **Quality > Capsule Edge Devices**. Copy the
one-time key immediately into the untracked .env.jetson file:

~~~text
OCEANMES_ENABLED=true
OCEANMES_BASE_URL=https://oceanmes.com
OCEANMES_EDGE_API_KEY=oce_edge_<one-time-secret>
OCEANMES_VERIFY_TLS=true
CAPSULE_EDGE_SOFTWARE_VERSION=<release-or-commit>
~~~

Plain HTTP is rejected by default. For an isolated LAN test only:

~~~text
OCEANMES_BASE_URL=http://<server-lan-address>:5001
OCEANMES_ALLOW_HTTP=true
~~~

The current Windows sandbox binds Waitress to 127.0.0.1, so it is not
reachable from a separate Jetson until the test server is deliberately exposed
on a LAN interface. Production should use HTTPS.

## Connectivity check

Inside the configured Jetson runtime/container:

~~~bash
python -m src.oceanmes config
~~~

The command prints device name, configuration version, production line, and
room. It never prints the bearer key.

## Retry ownership

The HTTP client performs no hidden POST retries. The future durable outbox will
retain one immutable UUID, canonical manifest, and JPEG and retry that same
content:

- network failure, HTTP 408/425/429, or 5xx: retry with exponential backoff;
- HTTP 200 duplicate acknowledgement: mark delivered;
- HTTP 201 new acknowledgement: mark delivered;
- HTTP 400/401/403/409/413/415/422: quarantine and surface for operator action.

This preserves the server's idempotency contract and prevents a retry from
silently becoming a new inspection.
