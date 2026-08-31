# Production readiness

Audit date: 2026-08-28

## Current assessment

The codebase is suitable for controlled development and supervised trials on a trusted Jetson network. It is not yet ready for an untrusted network or unattended production actuation. The immediate correctness and fail-safe defects identified in the audit have been fixed, but the deferred controls below are still required before a production release.

This document records recommendations only; it does not add product functionality.

## Issues addressed in this hardening pass

- Replaced unsupported Ultralytics `quantize` inference/export arguments with the supported `half` option and added CLI-path regressions.
- Added bounded, extra-field-forbidding server validation for settings updates and matching runtime validation for environment-derived settings.
- Required recent frames from every camera before inference readiness and before solenoid discharge.
- Changed the container health check from “`/stats` returned HTTP” to “the model exists and every configured camera has a recent running frame.”
- Made worker restart fail visibly when old capture/preview threads do not stop within their timeouts.
- Removed a hard-coded root partition UUID from IMX219 boot provisioning; the script now copies the current default entry's boot arguments.
- Preserved unrelated device-tree overlay lines when configuring a camera boot entry.
- Replaced a checked-in workstation-specific dataset path with a repository-relative path.
- Added one exact shared application dependency lock for local Python and Docker, with build/test-time drift verification and explicit Jetson platform exceptions.
- Expanded hardware-independent tests around precision arguments, settings bounds, health, counting fallbacks, dataset preparation, frame freshness, actuator gating, camera behavior, and worker lifecycle.

## Recommended next steps

### Priority 0: required before networked or unattended operation

1. Put the UI and API behind authentication, authorization, and TLS. Until then, bind to a management interface or firewall port 8000 to a trusted operator subnet. Settings and stop routes are currently unauthenticated.
2. Add independent hardware safety: a watchdog or monostable timeout, emergency stop, correctly rated isolated drivers, flyback protection, default-off pull states, and an interlock that does not depend on Python, Linux, or I2C remaining healthy.
3. Create a hardware-in-the-loop acceptance procedure that verifies polarity, boot-time output state, process crash, power loss, camera disconnect, stale frames, I2C errors, repeated restart, and emergency-stop behavior with valves physically disconnected first.
4. Define a release acceptance threshold on a held-out, representative production dataset. Record per-class precision/recall, confusion cases, throughput, frame latency, maximum stale-frame time, and lighting/camera tolerances.

### Priority 1: release engineering and operability

1. Add continuous integration for supported Python versions with unit tests, formatting/linting, type checks, coverage reporting, shell syntax checks, and a container build/config validation job.
2. Extend the exact shared direct pins into fully resolved transitive locks per target platform. Pin the Jetson base image by digest, retain the NVIDIA-provided CUDA PyTorch packages, generate an SBOM, and run dependency/container vulnerability scans.
3. Add structured logs, metrics, and alerts for model load, camera reconnects, inference latency, frame age, queue drops, health transitions, actuator phase, I2C errors, and restart failures. Avoid exposing images or sensitive paths in logs by default.
4. Formalize worker lifecycle state and serialize settings changes so concurrent requests cannot overlap restarts. Add integration tests for startup failure, shutdown timeout, malformed JSON, repeated updates, and unavailable models/cameras.
5. Run the service as a non-root user with a read-only root filesystem and explicit writable mounts. Pass only device nodes that are enabled and required; do not expose solenoid I2C access when actuation is disabled.

### Priority 2: reproducibility and maintainability

1. Version datasets and model artifacts with hashes, training configuration, source revision, metrics, calibration data, and a documented rollback procedure. Verify a model checksum and task/input shape before deployment.
2. Group related images before train/validation/test splitting to prevent near-duplicate or same-session leakage. Preserve a frozen golden evaluation set that dataset preparation cannot reshuffle.
3. Document and test backup/restore for Jetson boot configuration. Add fixture-based tests for both provisioning scripts using representative `extlinux.conf` layouts before any broader device rollout.
4. Establish supported JetPack/L4T, Python, CUDA, TensorRT, camera, and expander combinations. Remove obsolete hardware scripts or isolate them by platform once the deployed hardware matrix is final.
5. Add an operations runbook covering install, model promotion, rollback, health interpretation, log collection, calibration, camera replacement, and safe actuator disablement.

## Proposed production acceptance gates

A release should not be labeled production-ready until all of these are demonstrated:

- The unit, integration, container, and target hardware suites pass from a clean checkout in CI/release automation.
- The deployed model meets agreed quality and performance thresholds on a frozen evaluation set and target hardware.
- A stale or disconnected camera cannot cause discharge, including during startup, restart, and process failure.
- Independent hardware removes actuator power after the defined maximum-on interval even if the application hangs.
- Unauthorized clients cannot read video/stats or invoke settings/stop operations.
- Images, dependencies, model weights, and deployment configuration are immutable, versioned, scanned, and recoverable.
- Operators have tested monitoring, alerting, emergency stop, backup, and rollback procedures.

## Known verification limits of this audit

The automated tests run without Jetson cameras or an I2C expander. They validate control decisions and software behavior with fakes, not electrical behavior. No live model quality benchmark, TensorRT build, Docker image build, camera capture, autofocus run, boot overlay installation, or energized-solenoid test is implied by a passing unit suite.

The existing development venv is not itself a release artifact: `pip check` reports missing transitive metadata dependencies (`portalocker`, `cffi`, and `nvidia-cublas-cu12`) plus an unsupported CUDA package. The test wrapper addresses the observed cuDSS loader path only. Rebuild and lock clean desktop and Jetson environments as a release-engineering task; do not promote this venv or replace NVIDIA's Jetson PyTorch wheel in place.
